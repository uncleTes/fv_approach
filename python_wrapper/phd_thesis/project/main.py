# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import ConfigParser
import copy
import ExactSolution2D as ExactSolution2D
import Laplacian02 as Laplacian
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
from   petsc4py import PETSc
import time
import utilities
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

    edges = utilities.get_list_from_string(config.get("PABLO", "Edges"), 
                                           ","                         , 
                                           False)
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

    # Octants for grid.
    oct_f_g = [pow(2**dimension, ref) for ref in refinements]

    # Total number of octants on the totality of the grids.
    tot_oct = sum(oct_f_g)

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

    # Particles interaction.
    p_inter = config.getboolean("PROBLEM", "ParticlesInteraction")
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
        # Original points.
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
        # Matrix of tranformation coefficients from logical to physical.
        t_coeffs = utilities.p_t_coeffs(dim           , # Problem's dimension
                                        or_points     , # Original points
                                        t_points[grid], # Mapped points
                                        logger        , # Logger
                                        log_file)       # Log file
        # Adjoint matrix of tranformation coefficients from physical to logical.
        t_coeffs_adj = utilities.p_t_coeffs_adj(dim     , # Problem's dimension
                                                t_coeffs, # Transformation coeffs 
                                                logger  , # Logger
                                                log_file) # Log file
    
        trans_dictionary.update({grid : t_coeffs})
        trans_adj_dictionary.update({grid : t_coeffs_adj})

    return (trans_dictionary, trans_adj_dictionary)
    
# ------------------------------------------------------------------------------
def set_comm_dict(n_grids  ,
                  proc_grid,
                  comm_l):
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

    comm_dictionary = {}
    comm_dictionary.update({"edge" : ed})
    comm_dictionary.update({"communicator" : comm_l})
    comm_dictionary.update({"world communicator" : comm_w})
    comm_dictionary.update({"octants for grids" : oct_f_g})
    comm_dictionary.update({"total octants number" : tot_oct})
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
    comm_dictionary.update({"particles interaction" : p_inter})
    comm_dictionary.update({"log file" : log_file})
    comm_dictionary.update({"transformed points" : t_points})

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
    if not proc_grid:
        n_intercomms = n_grids - 1
        grids_to_connect = range(0, n_grids)
        grids_to_connect.remove(proc_grid)
    else:
        if not p_inter:
            n_intercomms = 1
            grids_to_connect = [0]
        else:
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
    pablo_log_file = "./log/" + comm_name + ".log"
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
   
    pablo.load_balance()
    pablo.update_connectivity()
    pablo.update_ghosts_connectivity()

    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    centers = numpy.empty([n_octs, dimension])
    
    for i in xrange(0, n_octs):
        centers[i, :] = pablo.get_center(i)[:dimension]

    return pablo, centers
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
    exact_solution = ExactSolution2D.ExactSolution2D(comm_dictionary)

    t_coeffs = numpy.array(None)
    t_coeffs_adj = numpy.array(None)
    
    trans_dictionary, trans_adj_dictionary = set_trans_dicts(n_grids  ,
                                                             dimension,
                                                             logger   ,
                                                             log_file)
    t_coeffs = trans_dictionary[proc_grid]
    t_coeffs_adj = trans_adj_dictionary[proc_grid]
    laplacian.init_trans_dict(trans_dictionary)
    laplacian.init_trans_adj_dict(trans_adj_dictionary)
    (d_nnz, o_nnz) = laplacian.create_mask()
    laplacian.init_sol()

    laplacian.init_mat((d_nnz, o_nnz))
    not_penalized_centers = laplacian.not_pen_centers
    # Physical centers.
    p_centers = [utilities.apply_persp_trans(dimension,
                                             center   , 
                                             t_coeffs , 
                                             logger   , 
                                             log_file)\
                 for center in not_penalized_centers]
    # Numpy physical centers.
    # Numpy's \".asarray()\" or \".array()\" function? Checkout this link:
    # http://stackoverflow.com/questions/14415741/numpy-array-vs-asarray
    # If the original array is a Python list, then use the second function,
    # because it will be always needed a copy of the object.
    n_p_centers = numpy.array(p_centers)

    exact_solution.e_sol(n_p_centers[:, 0]                              , 
                         n_p_centers[:, 1]                              ,
                         n_p_centers[:, 2] if (dimension == 3) else None,
                         numpy.array(None)                              ,
                         use_mapping = False)
    exact_solution.e_s_der(n_p_centers[:, 0], 
                           n_p_centers[:, 1],
                           n_p_centers[:, 2] if (dimension == 3) else None)
    laplacian.init_rhs(exact_solution.s_der)
    laplacian.set_b_c()
    laplacian.update_values(intercomm_dictionary)
    laplacian.solve()
    (norm_inf, norm_L2) = laplacian.evaluate_norms(exact_solution.sol,
                                                   laplacian.sol.getArray())
    msg = utilities.join_strings("process "               ,
                                 "%d " % comm_w.Get_rank(),
                                 "(%e, %e)" % (norm_inf, norm_L2))
                                 #str((norm_inf, norm_L2)))
    print(msg) 
    interpolate_sol = laplacian.reset_partially_solution()
    p_centers = [utilities.apply_persp_trans(dimension,
                                             center   , 
                                             t_coeffs , 
                                             logger   ,
                                             log_file)\
                 for center in centers]
    n_p_centers = numpy.array(p_centers)
    exact_solution.e_sol(n_p_centers[:, 0]                              , 
                         n_p_centers[:, 1]                              ,
                         n_p_centers[:, 2] if (dimension == 3) else None,
                         numpy.array(None)                              ,
                         use_mapping = False)
    data_to_save = numpy.array([exact_solution.sol,
                                interpolate_sol.getArray()])

    return (data_to_save, t_coeffs)
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

    comm_dictionary = set_comm_dict(n_grids  ,
                                    proc_grid,
                               	    comm_l)

    pablo, centers = set_octree(comm_l,
                                proc_grid)

    comm_dictionary.update({"octree" : pablo})
    comm_dictionary.update({"grid processes" : procs_l_lists[proc_grid]})
    # \"data_to_save\" = evaluated and exact solution;
    # \"trans_coeff\" = matrix containing perspective transformation's 
    # coefficients.
    data_to_save, trans_coeff = compute(comm_dictionary     ,
                                        intercomm_dictionary,
                                        proc_grid           ,
                                        centers             ,
                                        logger)

    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    #(geo_nodes, ghost_geo_nodes) = pablo.apply_persp_trans(dimension  ,
    #                                                       trans_coeff, 
    #                                                       logger     , 
    #                                                       log_file)
    geo_nodes = pablo.apply_persp_trans(dimension  ,
                                        trans_coeff, 
                                        logger     , 
                                        log_file)

    vtk = my_class_vtk.Py_My_Class_VTK(data_to_save            , # Data
                                       pablo                   , # Octree
                                       "./data/"               , # Dir
                                       "laplacian_" + comm_name, # Name
                                       "ascii"                 , # Type
                                       n_octs                  , # Ncells
                                       n_nodes                 , # Nnodes
                                       (2**dimension) * n_octs)  # (Nnodes * 
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
