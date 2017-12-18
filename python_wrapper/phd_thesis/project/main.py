# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import ConfigParser
import copy
import ExactSolution2D as ExactSolution2D
import Laplacian02fv as Laplacian
from   mpi4py import MPI
import my_class_vtk
import my_pablo_uniform
import numpy
import os
# http://sbrisard.github.io/posts/20130904-First_things_first_import_petsc4py_correctly.html
import sys
import petsc4py
# https://pythonhosted.org/petsc4py/apiref/petsc4py-module.html
petsc4py.init(sys.argv)
from petsc4py import PETSc
import time
import utilities
import pdb
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# http://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class ParsingFileException(Exception):
    """Class which derives from \"Exception\" which is raised when something
       with the config file is wrong."""
# ------------------------------------------------------------------------------

config_file = "./config/PABLO.ini"
log_file = "./log/Laplacian.log"
# Initialize the parser for the configuration file and read it.
config = ConfigParser.ConfigParser()
files_list = config.read(config_file)
# The previous "ConfigParser.read()" returns a list of file correctly read.
if len(files_list) == 0:
    msg = utilities.join_strings("Unable to read configuration file \"",
                                 config_file                           ,
                                 "\"."                                 ,
                                 # https://docs.python.org/3/library/os.html
                                 os.linesep                            ,
                                 "Program exited.")                
    print(msg)
    sys.exit(1)
# If some options or sections are not present, then the corresponding exception
# is catched, printed and the program exits.
try:
    n_grids = config.getint("PABLO", 
                            "NumberOfGrids")
    # Each grid will be anchored on (0, 0, 0).
    anchors = []
    anchor = [0, 0, 0]
    for i in xrange(0, n_grids):
        anchors.append(anchor)    

    #edges = utilities.get_list_from_string(config.get("PABLO", "Edges"),
    #                                       ","                         ,
    #                                       False)
    # Computational domain is (0.0, 0.0, 0.0) + 1.0 (square of unite edge,
    # anchored in the origin.
    edges = [1.0] * n_grids
    # \"t_points\" stands for \"transformed points\" because the mapping is done
    # between the computational domain (0, 0, 0) + edge and the physical one
    # given by "\GridPoints\".
    t_points = utilities.get_lists_from_string(config.get("PABLO",
                                                          "GridPoints"),
                                               ";"                     ,
                                               ","                     ,
                                               False)

    refinements = utilities.get_list_from_string(config.get("PABLO", 
                                                            "Refinements"), 
                                                 ",")
    dimension = config.getint("PROBLEM", "Dimension")

    assert (len(anchors) == n_grids)
    assert (len(edges) == n_grids)
    assert (len(refinements) == n_grids)
    # The form \"not anchors\" give us the possibility to check if \"anchors\"
    # is neither \"None\" or empty.
    # http://stackoverflow.com/questions/53513/best-way-to-check-if-a-list-is-empty
    if ((not anchors) or
        (not edges)   or
        (not refinements)):
        raise ParsingFileException

    # Log infos to log file or not.
    to_log = config.getboolean("PROBLEM", "Log")
except (ConfigParser.NoOptionError , 
        ConfigParser.NoSectionError,
        ParsingFileException       ,
        AssertionError):
    exc_info = str(sys.exc_info()[1])
    msg = utilities.join_strings("Program exited. Problems with config file \"",
                                 config_file                                   ,
                                 "\": "                                        ,
                                 exc_info                                      ,
                                 ".")
    print(msg)
    sys.exit(1)
# List of names for the MPI intercommunicators.
comm_names = ["comm_" + str(j) for j in range(n_grids)]
# Initialize MPI.
comm_w = MPI.COMM_WORLD
rank_w = comm_w.Get_rank()

# ------------------------------------------------------------------------------
def set_trans_dicts(n_grids,
                    dim    ,
                    logger ,
                    log_file):
    trans_dictionary = {}
    trans_adj_dictionary = {}
    for grid in xrange(0, n_grids):
        # Original points (grids with anchors (0, 0, 0) and edges 1.0).
        or_points = [anchors[grid][0], 
                     anchors[grid][1], 
                     anchors[grid][2], # Left low anchor.
                     anchors[grid][0] + edges[grid], 
                     anchors[grid][1], 
                     anchors[grid][2], # Right low anchor.
                     anchors[grid][0] + edges[grid], 
                     anchors[grid][1] + edges[grid], 
                     anchors[grid][2], # Right high anchor.
                     anchors[grid][0], 
                     anchors[grid][1] + edges[grid], 
                     anchors[grid][2]] # Left high anchor.
        # Original points, \"numpy\" version.
        n_or_points = numpy.array(or_points)
        n_or_points = numpy.reshape(n_or_points, (4,3))
        n_t_points = numpy.array(t_points[grid])
        n_t_points = numpy.reshape(n_t_points, (4,3))
        alpha = numpy.zeros(shape = (4, ),
                            dtype = numpy.float64)
        beta = numpy.zeros(shape = (4, ),
                           dtype = numpy.float64)
        utilities.bil_mapping(n_t_points       ,
                              alpha            ,
                              beta             ,
                              for_pablo = False,
                              dim = 2)
        # Matrix of transformation coefficients from logical to physical.
        t_coeffs = utilities.p_t_coeffs(dim           , # Problem's dimension
                                        n_or_points   , # Original points
                                        n_t_points)     # Mapped points
        # Adjoint matrix of transformation coefficients from physical to logical.
        t_coeffs_adj = utilities.p_t_coeffs_adj(dim     , # Problem's dimension
                                                t_coeffs) # Transformation coeffs
    
        trans_dictionary.update({grid : (t_coeffs, alpha, beta)})
        trans_adj_dictionary.update({grid : t_coeffs_adj})

    return (trans_dictionary, trans_adj_dictionary)
    
# ------------------------------------------------------------------------------
def set_comm_dict(n_grids  ,
                  proc_grid,
                  comm_l   ,
                  octs_f_g):
    """Method which set a dictionary (\"comm_dictionary\") which is necessary 
       for the parallelized classes like \"ExactSolution2D\" or 
       \"Laplacian\".
       
       Arguments:
           n_grids (int) : number of grids present in the config file.
           proc_grids (int) : number telling which grid thw current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.

       Returns:
           a dictionary, previously setted."""

    # Edge's length for PABLO.
    ed = edges[proc_grid]
    # Total number of octants present in the problem.
    tot_oct = numpy.sum(octs_f_g)

    refinement = refinements[proc_grid]

    comm_dictionary = {}
    comm_dictionary.update({"edge" : ed})
    comm_dictionary.update({"communicator" : comm_l})
    comm_dictionary.update({"world communicator" : comm_w})
    comm_dictionary.update({"octants for grids" : octs_f_g})
    comm_dictionary.update({"total octants number" : tot_oct})
    comm_dictionary.update({"total number of grids" : n_grids})
    background_boundaries = [anchors[0][0], anchors[0][0] + edges[0],
                             anchors[0][1], anchors[0][1] + edges[0]]
    if (dimension == 3):
        background_boundaries.append(anchors[0][2])
        background_boundaries.append(anchors[0][2] + edges[0])

    comm_dictionary.update({"background boundaries" : background_boundaries})
    foreground_boundaries = []
    f_list = range(1, n_grids)
    # If we are on the foreground grids we save all the foreground 
    # boundaries except for the ones of the current process. Otherwise, if
    # we are on the background grid, we save all the foreground grids.
    if proc_grid:
        f_list.remove(proc_grid)
    # If there is no foreground grid, \"foreground_boundaries\" will be 
    # empty.
    if len(f_list) >= 1:
        for i in f_list:
            boundary = [anchors[i][0], anchors[i][0] + edges[i],
                        anchors[i][1], anchors[i][1] + edges[i]]
            if (dimension == 3):
                boundary.append(anchors[i][2])
                boundary.append(anchors[i][2] + edges[i])

            foreground_boundaries.append(boundary)

    comm_dictionary.update({"foreground boundaries" : 
                            foreground_boundaries})
    comm_dictionary.update({"process grid" : proc_grid})
    comm_dictionary.update({"dimension" : dimension})
    comm_dictionary.update({"to log" : to_log})
    comm_dictionary.update({"log file" : log_file})
    comm_dictionary.update({"transformed points" : t_points})
    comm_dictionary.update({"refinement" : refinement})

    return comm_dictionary
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def create_intercomms(n_grids      ,
                      proc_grid    ,
                      comm_l       ,
                      procs_l_lists,
                      logger       ,
                      intercomm_dict = {}):
    """Method which creates the \"MPI\" intercommumicators for the different
       grids.
       
       Arguments:
           n_grids (int) : number of grids present in the config file.
           proc_grids (int) : number telling which grid the current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.
           procs_l_lists (list[lists]) : list containing the lists of processes
                                         for each grid.
           logger (utilities.Logger) : logger needed to log the 
                                       intercommunicators created.
           intercomm_dict (dict) : dictionary filled with the intercommunicators
                                   created."""
    n_intercomms = n_grids - 1
    grids_to_connect = range(0, n_grids)
    grids_to_connect.remove(proc_grid)
    # Communicator local's name. 
    comm_l_n = comm_l.Get_name()
    for grid in grids_to_connect:
        # Remote grid.
        r_grid = grid
        # Local grid.
        l_grid = proc_grid
        # List index.
        l_index = None

        if (l_grid == 0):
            l_index  = str(l_grid) + str(r_grid)
        else:
            if (r_grid == 0):
                l_index = str(r_grid) + str(l_grid)
            else:
                if (l_grid % 2 == 1):
                    l_index = str(r_grid) + str(l_grid)
                    if ((r_grid % 2 == 1) and
                        (r_grid > l_grid)):
                            l_index = str(l_grid) + str(r_grid)
                else:
                    l_index = str(l_grid) + str(r_grid)
                    if ((r_grid % 2 == 0) and
                        (r_grid > l_grid)):
                        l_index = str(r_grid) + str(l_grid)
        
        l_index = int(l_index)
        # Remote peer communicator.
        r_peer_comm = procs_l_lists[r_grid][0]
        # Local peer communicator.
        l_peer_comm = 0
        # http://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node145.htm
                                            # Local leader (each 
                                            # intracommunicator has \"0\" as  
                                            # leader).
        intercomm = comm_l.Create_intercomm(l_peer_comm,
                                            # Peer communicator in common 
                                            # between intracommunicators.
                                            comm_w     ,
                                            # Remote leader (in the 
                                            # MPI_COMM_WORLD it will be the
                                            # first of each group).
                                            r_peer_comm,
                                            # \"Safe\" tag for communication 
                                            # between the two process 
                                            # leaders in the MPI_COMM_WORLD 
                                            # context.
                                            l_index)
        
        intercomm_dict.update({l_index : intercomm})
        msg = utilities.join_strings("Created intercomm \"comm_"  ,
                                     "%d" % l_index               ,
                                     "\" for local comm \""       ,
                                     comm_l_n                     ,
                                     "\" and peer communicator \"",
                                     "%d" % l_peer_comm           ,
                                     "\" with remote comm \"comm_",
                                     "%d" % r_grid                ,
                                     "\" and peer communicator \"",
                                     "%d" % r_peer_comm           ,
                                     "\".")
        logger.info(msg)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def set_octree(comm_l,
               proc_grid):
    """Method which set the \"PABLO\" for the current process.
    
       Arguments:
           proc_grids (int) : number telling which grid thw current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.
            
       Returns:
           pablo (class_para_tree.Py_Class_Para_Tree_D2) : the octree.
           centers (list[lists]) : list containing lists of the abscissa and 
                                   ordinate coordinates of the centers of the
                                   quadtree of the current process."""

    comm_name = comm_l.Get_name()
    refinement_levels = refinements[proc_grid]
    # Anchor node for PABLO.
    an = anchors[proc_grid]
    # Edge's length for PABLO.
    ed = edges[proc_grid]
    pablo_log_file = "./log/" + comm_name
    pablo = my_pablo_uniform.Py_My_Pablo_Uniform(an[0]         ,
                                                 an[1]         ,
                                                 an[2]         ,
                                                 ed            ,
                                                 dimension     , # Dim
                                                 20            , # Max level
                                                 pablo_log_file, # Logfile
                                                 comm_l)         # Comm

    pablo.set_balance(0, True)
    for iteration in xrange(0, refinement_levels):
        pablo.adapt_global_refine()
    # TODO: Remember to download new \"Bitpit\" to get the patch for the problem
    #       with the two calls of \"load_balance()\" with few octants. The pro-
    #       blem was the wrong computation of the intersections.
    #
    pablo.load_balance()
    pablo.compute_intersections()

    n_octs = pablo.get_num_octants()

    #if (not proc_grid):
    #    for octant in xrange(0, n_octs):
    #        center = pablo.get_center(octant)[: dimension]
    #        ref_cond_x = center[0] >= 0.0625 and center[0] <= 0.9375
    #        ref_cond_y = center[1] >= 0.0625 and center[1] <= 0.9375
    #        ref_cond_x_y = ref_cond_x and ref_cond_y
    #        if (ref_cond_x_y):
    #            pablo.set_marker(octant, 1)
    #if (not proc_grid):
    #    for octant in xrange(0, n_octs):
    #        center = pablo.get_center(octant)[: dimension]
    #        ref_cond_x = center[0] >= 0.1 and center[0] <= 0.9
    #        ref_cond_y = center[1] >= 0.1 and center[1] <= 0.9
    #        ref_cond_x_y = ref_cond_x and ref_cond_y
    #        if (ref_cond_x_y):
    #            pablo.set_marker(octant, 1)

    #if (proc_grid):
    #    for octant in xrange(0, n_octs):
    #        center  = pablo.get_center(octant)[: dimension]
    #        # Refinement condition on \"x\".
    #        ref_cond_x = (center[0] < 0.75) and (center[0] > 0.25)
    #        #ref_cond_x = False
    #        ref_cond_x_02 = (center[0] > 0.25)
    #        #ref_cond_x = True
    #        #ref_cond_x = 0
    #        ref_cond_x_03 = (numpy.abs(center[0] - 0.5) <= 0.1)
    #        # Refinement condition on \"y\".
    #        ref_cond_y = (center[1] < 0.75) and (center[1] > 0.25)
    #        ref_cond_y_02 = (center[1] > 0.25)
    #        ref_cond_y_03 = (numpy.abs(center[1] - 0.5) <= 0.1)
    #        #ref_cond_x_y = (numpy.sqrt(numpy.square(center[0] - 0.5) + \
    #        #                           numpy.square(center[1] - 0.5)) <= 0.5)
    #        ref_cond_x_y = (numpy.sqrt(numpy.square(center[0] - 0.5) + \
    #                                   numpy.square(center[1] - 0.5)) <= 0.35)
    #        # Refinement condition on \"y\".
    #        # TODO: for 3D cases, implement this condition.
    #        ref_cond_z = True if (dimension == 2) else \
    #                     True
    #        #if ((ref_cond_x and ref_cond_y and ref_cond_z)):
    #            #or (ref_cond_x_02 and ref_cond_y_02 and ref_cond_z)):
    #        #if (ref_cond_x_03 and ref_cond_y_03):
    #        if ref_cond_x_y:
    #                pablo.set_marker(octant, 1)

    pablo.adapt()
    pablo.load_balance()
    pablo.update_connectivity()
    # Computing new intersections.
    pablo.compute_intersections()

    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    centers = numpy.empty([n_octs, dimension])
    
    for i in xrange(0, n_octs):
        centers[i, :] = pablo.get_center(i)[:dimension]

    return pablo, centers
# ------------------------------------------------------------------------------
def write_norms(n_norm_inf ,
                n_norm_L2  ,
                comm_l     ,
                proc_grid  ,
                file_extension):
    comm_l_size = comm_l.Get_size()
    comm_l_name = comm_l.Get_name()
    n_result_inf = numpy.zeros(shape = (1, ),
                               dtype = numpy.float64)
    n_result_L2 = numpy.zeros(shape = (1, ),
                              dtype = numpy.float64)
    comm_l.Reduce(n_norm_inf  ,
                  n_result_inf,
                  MPI.SUM)
    comm_l.Reduce(n_norm_L2  ,
                  n_result_L2,
                  MPI.SUM)
    if (comm_l.Get_rank() == 0):
        norm_inf = n_result_inf[0] / comm_l_size
        norm_L2 = n_result_L2[0] / comm_l_size
        r_level = refinements[proc_grid]
        path_to_file = utilities.join_strings(os.getcwd(),
                                              "/data/"   ,
                                              comm_l_name,
                                              file_extension)
        file_exists = os.path.exists(path_to_file)
        open_mod = "w"
        if (file_exists):
            open_mod = "a"
        f_d = open(path_to_file,
                   open_mod)
        msg = "%d %e %e" % (r_level, norm_inf, norm_L2)
        f_d.write(msg)
        f_d.write("\n")
        f_d.close()
# ------------------------------------------------------------------------------

def compute(comm_dictionary     ,
            intercomm_dictionary,
            proc_grid           ,
            centers             ,
            logger):
    """Method which compute all the calculation for the laplacian, exact 
       solution and residuals.

       Arguments:
           comm_dictionary (dict) : dictionary containing useful data for each
                                    intra-communicator and grid.
           intercomm_dictionary (dict) : dictionary containing the 
                                         intercommunicators created.
           proc_grid (int) : grid of the current process.
           centers (list[lists]) : list containing lists of the centers of the
                                   quadtree contained in the current process.
           logger (utilities.Logger) : logger needed to log the 
                                       intercommunicators created.

       Returns:
           data_to_save (numpy.array) : array containings the data to be saved
                                        subsequently into the \"VTK\" file."""

    laplacian = Laplacian.Laplacian(comm_dictionary)

    t_coeffs = numpy.array(None)
    t_coeffs_adj = numpy.array(None)
    
    trans_dictionary, trans_adj_dictionary = set_trans_dicts(n_grids  ,
                                                             dimension,
                                                             logger   ,
                                                             log_file)
    t_coeffs = trans_dictionary[proc_grid][0]
    alpha = trans_dictionary[proc_grid][1]
    beta = trans_dictionary[proc_grid][2]
    t_coeffs_adj = trans_adj_dictionary[proc_grid]
    laplacian.init_trans_dict(trans_dictionary)
    laplacian.init_trans_adj_dict(trans_adj_dictionary)
    d_nnz, \
    o_nnz, \
    h_s = laplacian.create_mask()
    # Not penalized centers.
    n_p_cs = numpy.array(laplacian.not_pen_centers)
    # TODO: change implementation of \"exact_sol\" and \"exact_2nd_der\" to
    #       do not copy anymore into the inner \numpy\" array of zeros, but to
    #       pass directly the values evaluated.
    e_sol = utilities.exact_sol(n_p_cs,
                                alpha ,
                                beta)
    # Initial guess equal to exact solution.
    #laplacian.init_sol(e_sol)
    laplacian.init_sol()
    e_2nd_der = utilities.exact_2nd_der(n_p_cs,
                                        alpha ,
                                        beta)
    laplacian.init_rhs()
    laplacian.init_mat((d_nnz, o_nnz))
    laplacian.fill_mat_and_rhs()
    # \"Numpy\" determinants.
    dets = numpy.linalg.det(utilities.jacobians_bil_mapping(n_p_cs, alpha, beta))
    # Absolute values \"numpy\" determinants.
    a_dets = numpy.absolute(dets)
    h_s2 = numpy.power(h_s, 2)
    if (not proc_grid):
        laplacian.add_rhs(e_2nd_der * h_s2 * a_dets)
    else:
        laplacian.add_rhs(e_sol)
    laplacian.update_values(intercomm_dictionary)
    #laplacian.mat.view()
    #laplacian.rhs.view()
    laplacian.solve()

    comm_l = comm_dictionary["communicator"]

    w_n = lambda x : write_norms(x[0]     ,
                                 x[1]     ,
                                 comm_l   ,
                                 proc_grid,
                                 x[2])

    n_norm_inf, \
    n_norm_L2 = laplacian.evaluate_norms(e_sol                   ,
                                         laplacian.sol.getArray(),
                                         h_s                     ,
                                         l2 = False              ,
                                         r_n_d = True)
    w_n((n_norm_inf,
         n_norm_L2 ,
         "_errors.txt"))

    n_norm_inf, \
    n_norm_L2 = laplacian.evaluate_residual_norms(e_sol                   ,
                                                  h_s                     ,
                                                  petsc_size = True       ,
                                                  l2 = False              ,
                                                  r_n_d = True)
    w_n((n_norm_inf,
         n_norm_L2 ,
         "_residuals.txt"))
    n_norm_inf, \
    n_norm_L2 = laplacian.evaluate_residual_norms(e_sol - laplacian.sol.getArray(),
                                                  h_s                     ,
                                                  petsc_size = True       ,
                                                  l2 = False              ,
                                                  r_n_d = True            ,
                                                  sub_rhs = False)
    w_n((n_norm_inf,
         n_norm_L2 ,
         "_residuals_diff_sols.txt"))
    n_norm_inf, \
    n_norm_L2 = laplacian.evaluate_residual_norms(laplacian.sol.getArray()                   ,
                                                  h_s                     ,
                                                  petsc_size = True       ,
                                                  l2 = False              ,
                                                  r_n_d = True)
    w_n((n_norm_inf,
         n_norm_L2 ,
         "_residuals_comp_sol.txt"))

    n_norm_inf, \
    n_norm_L2 = laplacian.evaluate_norms(laplacian.f_nodes      ,
                                         laplacian.f_nodes_exact,
                                         laplacian.h_s_inter    ,
                                         l2 = False             ,
                                         r_n_d = True)
    w_n((n_norm_inf,
         n_norm_L2 ,
         "_f_internal_nodes.txt"))

    if (n_grids > 1):
        if (proc_grid):
            n_norm_inf, \
            n_norm_L2 = laplacian.evaluate_norms(laplacian.f_on_bord         ,
                                                 laplacian.f_exact_on_bord   ,
                                                 laplacian.h_s_inter_on_board,
                                                 l2 = False                  ,
                                                 r_n_d = True)
            w_n((n_norm_inf,
                 n_norm_L2 ,
                 "_f_borders.txt"))
            #print(laplacian.grad_exact_x.shape)
            #print(laplacian.grad_rec_x.shape)
            n_norm_inf, \
            n_norm_L2 , \
            grad_x_array = laplacian.evaluate_norms(laplacian.grad_rec_x    ,
                                                    laplacian.grad_exact_x  ,
                                                    laplacian.h_s_inter_grad,
                                                    l2 = False              ,
                                                    r_n_d = True            ,
                                                    r_n_array = True)
            w_n((n_norm_inf,
                 n_norm_L2 ,
                 "_grad_x.txt"))
            n_norm_inf, \
            n_norm_L2 , \
            grad_y_array = laplacian.evaluate_norms(laplacian.grad_rec_y    ,
                                                    laplacian.grad_exact_y  ,
                                                    laplacian.h_s_inter_grad,
                                                    l2 = False              ,
                                                    r_n_d = True            ,
                                                    r_n_array = True)
            w_n((n_norm_inf,
                 n_norm_L2 ,
                 "_grad_y.txt"))
        else:
            grad_x_array = laplacian.grad_rec_x
            grad_y_array = laplacian.grad_rec_y

    e_sol = utilities.exact_sol(centers,
                                alpha  ,
                                beta)
    interpolate_sol = laplacian.reset_partially_array(array_to_reset = "sol",
                                                      is_array = True       ,
                                                      vector_to_reset = None,
                                                      vector_temp = e_sol)
    interpolate_res = laplacian.reset_partially_array(array_to_reset = "res")
    #print(laplacian.residual.getArray().shape)
    interpolate_grad_x = laplacian.reset_partially_array(array_to_reset = "grad_x",
                                                         is_array = False         ,
                                                         vector_to_reset = grad_x_array)
    interpolate_grad_y = laplacian.reset_partially_array(array_to_reset = "grad_y",
                                                         is_array = False         ,
                                                         vector_to_reset = grad_y_array)
    #    print(interpolate_grad_x.getArray().shape)
    #else:
    #    print(grad_y_array.shape)
    data_to_save = numpy.array([e_sol                     ,
                                interpolate_sol.getArray(),
                                interpolate_res.getArray(),
                                interpolate_grad_x.getArray(),
                                interpolate_grad_y.getArray()])

    #print(data_to_save.shape)

    return (data_to_save, t_coeffs, alpha, beta)
# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():
    """Main function....yeah, the name is self explanatory."""

    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    procs_l_lists = utilities.chunk_list_ordered(procs_w_list,
                                                 n_grids)
    proc_grid = utilities.get_proc_grid(procs_l_lists,
                                        comm_w.Get_rank())
    group_l = group_w.Incl(procs_l_lists[proc_grid])
    # Creating differents MPI intracommunicators.
    comm_l = comm_w.Create(group_l)
    # Current intracommunicator's name.
    comm_name = comm_names[proc_grid]
    comm_l.Set_name(comm_name)
    #Communicator local's name.
    comm_l_n = comm_l.Get_name()
    #Communicator local's rank.
    comm_l_r = comm_l.Get_rank()
    #Communicator global's name.
    comm_w_n = comm_w.Get_name()
    msg = utilities.join_strings("Started function for local comm \"",
                                 comm_l_n                            ,
                                 "\" and world comm \""              ,
                                 comm_w_n                            ,
                                 "\" and rank \""                    ,
                                 "%d" % comm_l_r                     ,
                                 "\".")            
    what_log = "debug"
    # If \"Log\" option in file \"PABLO.ini\" is \"False\" then the \"what_log\" 
    # flag will be \"critical\", to log only messages with importance equal or 
    # greater than \"critical\". But, usually in the context of this project, 
    # debug flags are just "info\" and \"error\". 
    if (not to_log):
        what_log = "critical"
    logger = utilities.Logger(__name__, 
                              log_file,
                              what_log).logger
    logger.info(msg)
    
    # Creating differents MPI intercommunicators.
    # http://www.linux-mag.com/id/1412/
    # http://mpi4py.scipy.org/svn/mpi4py/mpi4py/tags/0.4.0/mpi/MPI.py
    intercomm_dictionary = {}

    if procs_w > 1:
        create_intercomms(n_grids      ,
                          proc_grid    ,
                          comm_l       ,
                          procs_l_lists,
                          logger       ,
                          intercomm_dictionary)
    pablo, centers = set_octree(comm_l,
                                proc_grid)

    n_octs = numpy.zeros(1,
                         numpy.int64)
    n_octs[0] = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    comm_w_s = comm_w.size
    comm_l_s = comm_l.size

    # Octant for \"MPI\" processes.
    octs_f_p = numpy.zeros(comm_l_s,
                           dtype = int)
    comm_l.Allgather(n_octs,
                     [octs_f_p, 1, MPI.INT64_T])
    # Local number of total octants (here \"local\" means that is for each
    # octree).
    l_tot_oct = numpy.zeros(1,
                            numpy.int64)

    # Octant for grids (here we wanto to store an array of size \"n_grids\" to
    # save the number of octants for each grid (not for each \"MPI\" process).
    octs_f_g = numpy.zeros(n_grids,
                           dtype = int)
    # Send counts. Here we store how many element are sent by each process in
    # the world communicator.
    s_counts = numpy.zeros(comm_w_s,
                           dtype = numpy.int64)
    # Displacements for each process in the world communicator.
    displs = numpy.zeros(comm_w_s,
                         dtype = numpy.int64)
    # Local displacement for each process in the local communicator.
    l_displ = numpy.zeros(1,
                          dtype = numpy.int64)
    # Send count for each process in the local communicator.
    l_s_count = numpy.zeros(1,
                            dtype = numpy.int64)

    if (comm_l.Get_rank() == 0):
        l_tot_oct[0] = numpy.sum(octs_f_p)
        l_displ[0] = proc_grid
        l_s_count[0] = 1
    # If the rank of the process in the local communicator is equal to \"0\",
    # then we want it to send globally the \"l_tot_count\" parameter, with a
    # displacement in the final array where we will receive the data
    # (\"octs_f_g\") equal to the number of the current grid. If the rank of the
    # process, otherwies, is different from \"0\", it will send nothing global-
    # ly.
    comm_w.Allgather(l_s_count,
                     [s_counts, 1, MPI.INT64_T])
    comm_w.Allgather(l_displ,
                     [displs, 1, MPI.INT64_T])

    if (comm_l.Get_rank()):
        l_tot_oct = numpy.zeros(0,
                                dtype = numpy.int64)
    comm_w.Allgatherv(l_tot_oct,
                      [octs_f_g, s_counts, displs, MPI.INT64_T])

    comm_dictionary = set_comm_dict(n_grids  ,
                                    proc_grid,
                                    comm_l   ,
                                    octs_f_g)

    comm_dictionary.update({"octree" : pablo})
    comm_dictionary.update({"grid processes" : procs_l_lists[proc_grid]})
    # \"data_to_save\" = evaluated and exact solution;
    # \"trans_coeff\" = matrix containing perspective transformation's 
    # coefficients.
    data_to_save, trans_coeff, alpha, beta = compute(comm_dictionary     ,
                                                     intercomm_dictionary,
                                                     proc_grid           ,
                                                     centers             ,
                                                     logger)
    #print(data_to_save.shape)

    #(geo_nodes, ghost_geo_nodes) = pablo.apply_persp_trans(dimension  ,
    #                                                       trans_coeff, 
    #                                                       logger     , 
    #                                                       log_file)
    geo_nodes = pablo.apply_persp_trans(dimension,
                                        alpha    ,
                                        beta)
    comm_w.Barrier()
    vtk = my_class_vtk.Py_My_Class_VTK(data_to_save            ,  # Data
                                       pablo                   ,  # Octree
                                       "./data/"               ,  # Dir
                                       "laplacian_" + comm_name,  # Name
                                       "ascii"                 ,  # Type
                                       n_octs[0]               ,  # Ncells
                                       n_nodes                 ,  # Nnodes
                                       (2**dimension) * n_octs[0])# (Nnodes *
                                                                  #  pow(2,dim))
    #vtk.apply_trans(geo_nodes, ghost_geo_nodes) 
    vtk.apply_trans(geo_nodes) 
    ## Add data to "vtk" object to be written later.
    vtk.add_data("evaluated", # Data
                 1          , # Data dim
                 "Float64"  , # Data type
                 "Cell"     , # Cell or Point
                 "ascii")     # File type
    vtk.add_data("exact"  , 
                 1        , 
                 "Float64", 
                 "Cell"   , 
                 "ascii")
    vtk.add_data("residual"  ,
                 1           ,
                 "Float64"   ,
                 "Cell"      ,
                 "ascii")
    vtk.add_data("grad_x"  ,
                 1           ,
                 "Float64"   ,
                 "Cell"      ,
                 "ascii")
    vtk.add_data("grad_y"  ,
                 1           ,
                 "Float64"   ,
                 "Cell"      ,
                 "ascii")
    # Call parallelization and writing onto file.
    vtk.print_vtk()

    msg = utilities.join_strings("Ended function for local comm \""  ,
                                 comm_l_n                            ,
                                 "\" and world comm \""              ,
                                 comm_w_n                            ,
                                 "\" and rank \""                    ,
                                 "%d" % comm_l_r                     ,
                                 "\".")            
    msg = "".join(msg)

    logger.info(msg)
# ------------------------------------------------------------------------------
    
if __name__ == "__main__":

    if rank_w == 0:
        msg = "STARTED LOG"
        logger = utilities.log_msg(msg, 
                                   log_file)
        msg = utilities.join_strings("NUMBER OF GRIDS: ",
                                     "%d." % n_grids)
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = utilities.join_strings("ANCHORS: " ,
                                     str(anchors),
                                     ".")
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = utilities.join_strings("EDGES: " ,
                                     str(edges),
                                     ".")
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = utilities.join_strings("REFINEMENT LEVELS: " ,
                                     str(refinements)      ,
                                     ".")
        utilities.log_msg(msg     ,
                          log_file,
                          logger)

    t_start = time.time()

    import cProfile
    # http://stackoverflow.com/questions/3898266/what-is-this-cprofile-result-telling-me-i-need-to-fix
    #cProfile.run('main()', sort='cumulative')
    main()
    #pdb.run('main()')

    comm_w.Barrier()

    if rank_w == 0:
        file_name = "multiple_PABLO.vtm"
        files_vtu = utilities.find_files_in_dir(".vtu", 
                                                "./data/")
    
        info_dictionary = {}
        info_dictionary.update({"vtu_files" : files_vtu})
        info_dictionary.update({"pablo_file_names" : comm_names})
        info_dictionary.update({"file_name" : file_name})
	info_dictionary.update({"directory" : "./data"})
    
        #write_vtk_multi_block_data_set(**info_dictionary)
        utilities.write_vtk_multi_block_data_set(info_dictionary)
    
        t_end = time.time()

        msg = "EXECUTION TIME: %f secs." % (t_end - t_start)
        file_msg = utilities.join_strings(str(t_end - t_start),
                                          "\n")
        f = open('benchmark.txt', 'a')
        f.write(file_msg)
        f.close()
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = "ENDED LOG"
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
