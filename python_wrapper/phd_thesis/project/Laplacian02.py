# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# A guide to analyzing Python performance:
# http://www.huyng.com/posts/python-performance-analysis/
# ------------------------------------IMPORT------------------------------------
import numbers
import math
import collections
import BaseClass2D
import ExactSolution2D
import numpy
# http://sbrisard.github.io/posts/20130904-First_things_first_import_petsc4py_correctly.html
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc 
from mpi4py import MPI
import utilities
import time
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Laplacian(BaseClass2D.BaseClass2D):   
    """Class which evaluates the laplacian onto a grid.
    
    Attributes:
        _comm (MPI.Intracomm) : intracommunicator which identify the
                                process where evaluate the laplacian.
        _octree (class_para_tree.Py_Class_Para_Tree_D2) : PABLO's ParaTree.
        _comm_w (MPI.Intracomm) : global intracommunicator.
        _over_l (boolean) : flag inndicating if we are in an overlapped or 
                            full immersed case.
        _f_bound (list of lists) : foreground boundaries (boundaries of the
                                   grids over the background ones).
        _b_bound (list of numbers) : background boundaries.
        _proc_g (int) : grid for which the current process is doing all the 
                        work.
        _N_oct (int) : total number of octants in the communicator.
        _n_oct (int) : local number of octants in the process.
        _edge (number) : length of the edge of the grid.
        _grid_l (list) : list of processes working on the current grid.
        _tot_oct (int) : total number of octants in the whole problem.
        _oct_f_g (list) : list of octants for each grid presents into the 
                          problem.
        _h (number): edge's length of the edge of the octants of the grid.
        _rank_w (int) : world communicator rank.
        _rank (int) : local communicator rank.
        _masked_oct_bg_g (int) : number of masked octants on the background 
                                 grid."""
    
    # --------------------------------------------------------------------------
    def __init__(self, 
                 kwargs = {}):
        """Initialization method for the \"Laplacian\" class.

        Arguments:
            kwargs (dictionary) : it must contains the following keys (in 
                                  addition to the ones of \"BaseClass2D\"):
                                  - \"edge\".
                                     
            Raises:
                AssertionError : if \"edge" is not greater than 0.0, then the 
                                 exception is raised and catched, launching an 
                                 \"MPI Abort\", launched also if attributes 
                                 \"f_bound\" or \"b_bound\" are \"None\"."""

        # http://stackoverflow.com/questions/19205916/how-to-call-base-classs-init-method-from-the-child-class
        super(Laplacian, self).__init__(kwargs)
        # If some arguments are not presents, function \"setdefault\" will set 
        # them to the default value.
        # Particles interaction.
        self._p_inter = kwargs.setdefault("particles interaction",
                                          False)
        # \"[[x_anchor, x_anchor + edge, 
        #     y_anchor, y_anchor + edge]...]\" = penalization boundaries (aka
        # foreground boundaries).
        self._f_bound = kwargs.setdefault("foreground boundaries",
                                          None)
        # \"[x_anchor, x_anchor + edge, 
        #    y_anchor, y_anchor + edge]\" = background boundaries.
        self._b_bound = kwargs.setdefault("background boundaries",
                                          None)
        # Checking existence of penalization boundaries and background 
        # boundaries. The construct \"if not\" is useful whether 
        # \"self._f_bound\" or \"self._b_bound\" are None or with len = 0.
        # TODO: sistema questo if, puoi anche far passare a tutte le griglie 
        # i \"self._f_bound\".
        if (not self._b_bound):
        #if ((not self._f_bound) or 
        #    (not self._b_bound)):
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Penalization or bakground boundaries or both are " + \
                        "not initialized. Please check your \"config file\"."
            self.log_msg(msg    , 
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1) 
        # The grid of the current process: process' grid.
        self._proc_g = kwargs["process grid"]
        # Total number of octants into the communicator.
        self._N_oct = self._octree.get_global_num_octants()
        # Local number of octants in the current process of the communicator.
        self._n_oct = self._octree.get_num_octants()
        # Length of the edge of the grid.
        self._edge = kwargs["edge"]
        self._grid_l = kwargs["grid processes"]
        try:
            assert self._edge > 0.0
        except AssertionError:
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Attribute \"self._edge\" equal or smaller than 0."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1)
        # Total number of octants presents in the problem.
        self._tot_oct = kwargs["total octants number"]
        # Number of octants for each grid. It is a list.
        self._oct_f_g = kwargs["octants for grids"]
        # Length of the edge of an octree.
        self._h = self._edge / numpy.sqrt(self._N_oct)
        # Getting the rank of the current process inside the world communicator
        # and inside the local one.
        self._rank_w = self._comm_w.Get_rank()
        self._rank = self._comm.Get_rank()
        t_points = kwargs.setdefault("transformed points", None)
        if (not t_points):
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Transformed points with mapping \"True\" are " + \
                        "not initialized. Please check your \"config file\"."
            self.log_msg(msg    , 
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1)
        # Transformed background.
        self._t_background = [] 
        # Transformed polygons.
        self._t_foregrounds = []
        for i, polygon in enumerate(t_points):
            # Temporary transformed points.
            t_t_points = []
            n_coordinates = len(polygon)
            #TODO: modify to do 3D!!
            for j in xrange(0, n_coordinates, 3):
                t_t_points.append([polygon[j], polygon[j + 1]])
                j += 2
            if (i == 0):
                self._t_background = t_t_points
            else:
                self._t_foregrounds.append(t_t_points)
        # Initializing exchanged structures.
        self.init_e_structures()
    # --------------------------------------------------------------------------

    # Returns the center of the face neighbour.
    # TODO: modify this function to be used in 3D case.
    def neighbour_centers(self   ,
                          # Centers.
                          cs     ,
                          # Edges or nodes.
                          es_o_ns,
                          # Values.
                          vs):
        """Function which returns the centers of neighbours, depending on 
           for which face we are interested into.
           
           Arguments:
               cs (tuple or list of tuple) : coordinates of the centers of 
                                             the current octree.
               es_o_ns (int between 1 and 2 or list) : numbers indicating if the 
                                                       neighbour is from edge or 
                                                       node.
               vs (int between 0 and 3 or list) : faces for which we are 
                                                  interested into knowing the 
                                                  neighbour's center.
                                            
           Returns:
               a tuple or a list containing the centers evaluated."""

        h = self._h
        centers = cs
        values = vs
        edges_or_nodes = es_o_ns        
        # Checking if passed arguments are lists or not. If not, we have to do
        # something.
        try:
            len(centers)
            len(values)
            len(edges_or_nodes)
        # \"TypeError: object of type 'int' has no len()\", so are no lists but
        # single elements.
        except TypeError:
            t_center, t_value, t_e_o_n = centers, values, edges_or_nodes
            centers, values, edges_or_nodes = ([] for i in range(0, 3))
            centers.append(t_center)
            values.append(t_value)
            edges_or_nodes.append(t_e_o_n)

        if ((len(values) != 1) and # TODO: Check if this first check is useless.
            (len(values) != len(centers))):
            msg = "\"MPI Abort\" called " 
            extra_msg = " Different length of \"edges\" (or \"nodes\") and " +\
                        "\"centers\"."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
            self._comm_w.Abort(1)

	# Evaluated centers.
        eval_centers = []
        # Face or node.
        for i, value in enumerate(values):
            (x_center, y_center) = centers[i]
            if not isinstance(value, numbers.Integral):
                value = int(math.ceil(e_o_n))
            try:
                # Python's comparison chaining idiom.
                assert 0 <= value <= 3
            except AssertionError:
                msg = "\"MPI Abort\" called " 
                extra_msg = " Faces numeration incorrect."
                self.log_msg(msg    ,
                             "error",
                             extra_msg)
                self._comm_w.Abort(1)
            # If no exception is raised, go far...
            else:
                edge = False
                node = False
                if edges_or_nodes[i] == 1:
                    edge = True
                elif edges_or_nodes[i] == 2:
                    node = True
                if ((value % 2) == 0):
		    if (value == 0):
                    	x_center -= h
                        if node:
                            y_center -= h
		    else:
                        if edge:
                    	    y_center -= h
                        if node:
                            x_center -= h
                            y_center += h
                else:
		    if (value == 1):
                    	x_center += h
                        if node:
                            y_center -= h
		    else:
                    	y_center += h
                        if node:
                            x_center += h
                            
                eval_centers.append((x_center, y_center))

        msg = "Evaluated centers of the neighbours"
        if len(centers) == 1:
            msg = "Evaluated center of the neighbour"
            eval_centers = eval_centers[0]
                
        self.log_msg(msg   ,
                     "info")

        return eval_centers
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Evaluate boundary conditions. 
    def eval_b_c(self   ,
                 centers,
                 f_o_n  ,
                 codim = None):
        """Method which evaluate boundary condition on one octree or more,
           depending by the number of the \"center\" passed by.
           
           Arguments:
               centers (tuple or list of tuple) : coordinates of the center/s
                                                  of the octree on the boundary.
               f_o_n (int between 0 and 3 or list of int) : the face or node of 
                                                            the current octree 
                                                            for which we are
                                                            interested
                                                            into knowing the 
                                                            neighbour's center.
                                                           
           Returns:
               boundary_values (int or list) : the evaluated boundary condition
                                               or a list of them.
               c_neighs (tuple or list of tuples) : the centers where evaluate
                                                    the boundary conditions."""

        edges_or_nodes = []
        just_one_neighbour = False
        proc_grid = self._proc_g
        dimension = self._dim

        # Current transformation matrix's dictionary.
        c_t_dict = self.get_trans(proc_grid)
        if (codim is None):
            for i in xrange(0, len(centers)):
                # Evaluating boundary conditions for edges.
                edges_or_nodes.append(1)
        else:
            edges_or_nodes = codim
        # Centers neighbours.
        c_neighs = self.neighbour_centers(centers       ,
                                          edges_or_nodes,
                                          f_o_n)
        # \"c_neighs\" is only a tuple, not a list.
        if not isinstance(c_neighs, list):
            just_one_neighbour = True
            c_neigh = c_neighs
            c_neighs = []
            c_neighs.append(c_neigh)
        x_s = [c_neigh[0] for c_neigh in c_neighs] 
        y_s = [c_neigh[1] for c_neigh in c_neighs]
        if (dimension == 3):
            z_s = [c_neigh[2] for c_neigh in c_neighs]
        else:
            z_s = None
        
        boundary_values = ExactSolution2D.ExactSolution2D.solution(x_s     , 
                                                   		   y_s     ,
                                                                   z_s     ,
                                                                   c_t_dict)

        msg = "Evaluated boundary conditions"
        if just_one_neighbour: 
            msg = "Evaluated boundary condition"
        self.log_msg(msg   ,
                     "info")

        return (boundary_values, c_neighs)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Get octants's ranges for processes, considering also the ones masked by
    # the foreground grids, whom PETSc does not count using 
    # \"getOwnershipRange\".
    def get_ranges(self):
        """Method which evaluate ranges of the octree, for the current process.
                                                           
           Returns:
               the evaluated octants' range."""

        grid = self._proc_g
        n_oct = self._n_oct
        o_ranges = self._b_mat.getOwnershipRange()
        is_background = True
        if grid:
            is_background = False
        # If we are on the background grid, we need to re-evaluate the ranges
        # of octants owned by each process, not simply adding the masked ones
        # (as we will do for the poreground grids), to the values given by PETSc
        # function \"getOwnershipRange\" on the matrix. The idea is to take as
        # start of the range the sum of all the octants owned by the previous
        # processes of the current process, while for the end of the 
        # take the same range of processes plus the current one, obviously 
        # subtracting the value \"1\", because the octants start from \"0\".
        if is_background:
            # Local rank.
            rank_l = self._rank
            # Range's start.
            r_start = 0
            # Range's end.
            r_end = self._s_counts[0] - 1
            # Octants for process.
            octs_f_process = self._s_counts
            for i in xrange(0, rank_l):
                r_start += octs_f_process[i]
                r_end += octs_f_process[i + 1]
            new_ranges = (r_start, r_end)
        else:
            # Masked octants
            masked_octs = self._masked_oct_bg_g
            new_ranges = (o_ranges[0] + masked_octs, 
                          o_ranges[1] + masked_octs)

        msg = "Evaluated octants' ranges"
        extra_msg = "with ranges " + str(new_ranges) 
        self.log_msg(msg   ,
                     "info",
                    extra_msg)

        return new_ranges
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Set boundary conditions.
    def set_b_c(self):
        """Method which set boundary conditions for the different grids."""
    
	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        grid = self._proc_g
        n_oct = self._n_oct
        octree = self._octree
        nfaces = octree.get_n_faces()
        face_node = octree.get_face_node()
        h = self._h
        h_half = 0.5 * h
        h2 = h * h
        h2_inv_neg = -1.0 / h2
        is_background = False if (grid) else True
        o_ranges = self.get_ranges()
        dimension = self._dim
        t_b_centers = []
        # Current transformation matrix's dictionary.
        c_t_dict = self.get_trans(grid)
        # Current transformation adjoint matrix's dictionary.
        c_t_adj_dict = self.get_trans_adj(grid) 
        # Background transformation matrix adjoint's dictionary.
        b_t_adj_dict = self.get_trans_adj(0)
        # Transformed background.
        t_background = self._t_background

        b_indices, b_values = ([] for i in range(0, 2))# Boundary indices/values
        b_centers, b_f_o_n = ([] for i in range(0, 2)) # Boundary centers/faces
        b_codim = [] # Boundary codimensions

        # Code hoisting.
        mask_octant = self.mask_octant
        get_octant = octree.get_octant
        get_center = octree.get_center
        get_bound = octree.get_bound
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygon = utilities.is_point_inside_polygon                            
                              
        for octant in xrange(0, n_oct):
            # Global index of the current local octant \"octant\".
            g_octant = o_ranges[0] + octant
            m_g_octant = mask_octant(g_octant)
            # Check if the octant is not penalized.
            if (m_g_octant != -1):
                py_oct = get_octant(octant)
                center  = get_center(octant)[: dimension]

                # Lambda function.
                g_b = lambda x : get_bound(py_oct, x)

                # Nodes yet seen. Using Python \"sets\" to avoid duplicates.
                n_y_s = set()
                for face in xrange(0, nfaces):
                    # If we have an edge on the boundary.
                    if (g_b(face)):
                        b_indices.append(m_g_octant)
                        b_f_o_n.append(face)
                        b_centers.append(center)
                        b_codim.append(1)
                        n_y_s.update(face_node[face][0 : 2])
                for node in n_y_s:
                    b_indices.append(m_g_octant)
                    b_f_o_n.append(node)
                    b_centers.append(center)
                    b_codim.append(2)
        
        (b_values, c_neighs) = self.eval_b_c(b_centers,
                                             b_f_o_n  ,
                                             b_codim)
        
        prev_b_center = None
        prev_t_b_center = None
        for i, b_center in enumerate(b_centers):
            same_center = False
            # They are equal.
            # \"cmp()\" removed from python 3.x...better using \"==\", is
            # also more readable as returned value.
            # http://stackoverflow.com/questions/18674988/comparison-of-list-using-cmp-or
            #if (cmp(b_center, prev_b_center) == 0):
            if (b_center == prev_b_center):
                t_b_center = prev_t_b_center
                same_center = True
            if (not same_center):
                prev_b_center = b_center
                t_b_center =  apply_persp_trans(dimension, 
                                                b_center , 
                                                c_t_dict ,
                                                logger   ,  
                                                log_file)[: dimension]
                prev_t_b_center = t_b_center
                
            t_b_centers.append(t_b_center)
                    

        # Grids not of the background: equal to number >= 1.
        if (grid):
            for i, center in enumerate(c_neighs):
                check = False
                # Check if foreground grid is inside the background one.
                threshold = 0.0
                t_center =  apply_persp_trans(dimension, 
                                              center   , 
                                              c_t_dict ,
                                              logger   ,  
                                              log_file)[: dimension]
                check = is_point_inside_polygon(t_center    ,
                                                t_background,
                                                logger      ,
                                                log_file    ,
                                                threshold)
                if (check):
                    # Can't use list as dictionary's keys.
                    # http://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python
                    # https://wiki.python.org/moin/DictionaryKeys
                    key = (grid        , # Grid to which the index belongs to
                           b_indices[i], # Masked global index of the octant
                           b_f_o_n[i]  , # Boundary face
                           b_codim[i]  , # Boundary codimension
                           h)            # Edge's length
                    # We store the center of the cell on the boundary.
                    # TODO: why we store also \"center\" and not just 
                    #       \"b_centers[i]\"? Think about it.
                    t_value = tuple(center[: dimension])
                    t_value = t_value + tuple(b_centers[i][: dimension])
                    n_mask = 43 - len(t_value)
                    stencil = (t_value + ((-1,) * (n_mask))\
                               if (self._p_inter) else t_value)
                    self._edl.update({key : stencil})
                    # The new corresponding value inside \"b_values\" would be
                    # \"0.0\", because the boundary value is given by the 
                    # coefficients of the bilinear operator in the \"extension\"
                    # matrix.
                    b_values[i] = 0.0
        
        metric_coefficients = utilities.metric_coefficients(dimension,
                                                            t_b_centers,
                                                            c_t_adj_dict,
                                                            logger,
                                                            log_file)
        n_values = len(b_values)
        # Temporary multipliers.
        #t_ms = [0] * n_values
        # \"adj_matrix[0][0]\"...

        for i in xrange(0, n_values):
            i_metric_coefficients = metric_coefficients[:, i]
            codim = b_codim[i]
            index = b_f_o_n[i]
            A00 = i_metric_coefficients[0] 
            A10 = i_metric_coefficients[1]
            A01 = i_metric_coefficients[2]
            A11 = i_metric_coefficients[3]
            ds2_epsilon_x = i_metric_coefficients[4]
            ds2_epsilon_y = i_metric_coefficients[5]
            ds2_nu_x = i_metric_coefficients[6]
            ds2_nu_y = i_metric_coefficients[7]
            A002 = numpy.square(A00)
            A102 = numpy.square(A10)
            A012 = numpy.square(A01)
            A112 = numpy.square(A11)
            if (codim == 1):
                # Temporary multiplier.
                t_m = (A002 + A102) if (index < 2) else \
                      (A012 + A112)
                # Temporary multiplier 02.
                t_m02 = (ds2_epsilon_x + ds2_epsilon_y) if \
                        (index < 2) else (ds2_nu_x + ds2_nu_y)
                t_m02 *= h_half
                t_m += t_m02 if ((index % 2) == 1) else \
                       (-1.0 * t_m02)   
            # Codim == 2, so we are speaking about nodes and not edges.
            else:
                t_m = (A00 * A01) + (A10 * A11)
                t_m *= 0.5 if ((index == 0) or (index == 3)) \
                           else -0.5

            # TODO: Sum coefficients^2 for 3D.
            if (dimension == 3):
                pass

            t_m = h2_inv_neg * t_m
            b_values[i] = b_values[i] * t_m
            # The three following lines are just syntactic sugar to express
            # some python's capabilities. But the previous one line is just
            # faster. Tested personally on a list of 10000.
            #t_ms[i] = t_m
        #b_values = map(lambda pair : (pair[0] * pair[1]), 
        #               zip(b_values, t_ms))
   
        insert_mode = PETSc.InsertMode.ADD_VALUES
        self._rhs.setValues(b_indices,
                            b_values ,
                            insert_mode)

        self.assembly_petsc_struct("rhs")
        
        msg = "Set boundary conditions"
        extra_msg = "of grid \"" + str(self._proc_g) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Apply mask to the octants, not being considering the octants of the 
    # background grid covered by the ones of the foreground meshes.
    def mask_octant(self, 
                    g_octant):
        """Method which evaluate the global index of the octant, considering
           the masked ones, dued to the grids' overposition.
           
           Arguments:
               g_octant (int) : global index of the octant.

           Returns:
               m_g_octant (int) : masked global index of the octant."""

        grid = self._proc_g
        if grid:
            # Masked global octant.
            m_g_octant = g_octant - self._masked_oct_bg_g
        else:
            m_g_octant = self._ngn[g_octant]
        
        return m_g_octant
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    def check_neighbour(self         ,
                        codim        ,
                        f_o_n        ,
                        neighs       ,
                        ghosts       ,
                        octant       ,
                        o_count      ,
                        d_count      ,
                        # Stencil's index.
                        s_i          ,
                        h            ,
                        key          ,
                        octree       ,
                        is_penalized ,
                        is_background,
                        # Number of polygon to which the neigbour became to.
                        n_polygon_n  ,
                        yet_masked = False):
        logger = self.logger
        log_file = logger.handlers[0].baseFilename
        dimension = self._dim

        # Code hoisting.
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons

        if (yet_masked):
            oct_offset = o_count
            o_ranges = d_count
        # Check to know if a neighbour of an octant is penalized.
        is_n_penalized = False
        (neighs, ghosts) = octree.find_neighbours(octant, 
                                                  f_o_n , 
                                                  codim , 
                                                  neighs, 
                                                  ghosts)
        # If we are outside the current octree...
        # Empty lists in python are \"False\".
        #TODO: check if this \"if\" is useful or not.
        if ((codim == 2) and (not ghosts)):
            return (d_count, o_count, s_i) if (not yet_masked) else \
                   (None, None, None)
        # ...else...
        if not ghosts[0]:
            if (yet_masked):
                n_center = octree.get_center(neighs[0])[:2]
                index = neighs[0] + o_ranges[0]
                # Masked index.
                m_index = self.mask_octant(index)
            else:
                index = octree.get_global_idx(neighs[0])
                n_center = self._octree.get_center(neighs[0])[:2]
        else:
            index = self._octree.get_ghost_global_idx(neighs[0])
            py_ghost_oct = self._octree.get_ghost_octant(neighs[0])
            n_center = self._octree.get_center(py_ghost_oct, 
                                               True)[:2]
            if (yet_masked):
                m_index = self.mask_octant(index)
                m_index = m_index + oct_offset
        if is_background:
            is_n_penalized = True
            threshold = 0.0
            t_foregrounds = self._t_foregrounds
            # Current transformation matrix's dictionary.
            c_t_dict = self.get_trans(0)
            oct_corners = utilities.get_corners_from_center(n_center,
                                                            h)
            for i, corner in enumerate(oct_corners):
                is_corner_penalized = False
                corner = apply_persp_trans(dimension, 
                                           corner   , 
                                           c_t_dict ,
                                           logger   ,  
                                           log_file)[: dimension]
                (is_corner_penalized,
                 n_polygon) = is_point_inside_polygons(corner       ,
                                                       t_foregrounds,
                                                       logger       ,
                                                       log_file     ,
                                                       threshold)
                if (not is_corner_penalized):
                    is_n_penalized = False
                    break
        if (not yet_masked):
            if not is_penalized:
                if is_n_penalized:
                    # Being the neighbour penalized, it means that it 
                    # will be substituted by 9 octant being part of 
                    # the foreground grids, so being on the non diagonal
                    # part of the grid.
                    o_count += 9
                else:
                    if ghosts[0]:
                        o_count += 1
                    else:
                        d_count += 1
            else:
                if not is_n_penalized:
                    stencil = self._edl.get(key)
                    stencil[s_i] = index
                    stencil[s_i + 1], stencil[s_i + 2] = n_center
                    stencil[s_i + 3] = codim
                    stencil[s_i + 4] = f_o_n
                    self._edl[key] = stencil
                    s_i += 5

            extra_msg = ""

        else:
            extra_msg = "yet masked"

        msg = "Checked neighbour for "               + \
              ("edge " if (codim == 1) else "node ") + \
              str(index)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
        
        return (d_count, o_count, s_i) if (not yet_masked) else \
               (m_index, is_n_penalized, n_center)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Creates masking system for the octants of the background grid covered by 
    # the foreground meshes, and also determines the number of non zero elements
    # to allocate for each row in the system's matrix. Be aware of that, for the
    # moment, this last count is exact for the background grid but for the 
    # foreground ones it consider the worst case (for the two levels gap).
    #@profile
    def create_mask(self):
        """Method which creates the new octants' numerations and initialize non
           zero elements' number for row in the matrix of the system.
           
           Returns:
               (d_nnz, o_nnz) (tuple) : two lists containting the diagonal
                                        block's and non diagonal block's number
                                        of non zero elements."""

        logger = self.logger
        log_file = logger.handlers[0].baseFilename
        grid = self._proc_g
        n_oct = self._n_oct
        octree = self._octree
        nfaces = octree.get_n_faces()
        face_node = octree.get_face_node()
        h = self._h
        dimension = self._dim
        h2 = h * h
        is_background = True if (not grid) else False
        # Lists containing number of non zero elements for diagonal and non
        # diagonal part of the coefficients matrix, for row. 
        d_nnz, o_nnz = ([] for i in range(0, 2))
        new_oct_count = 0
        # \"range\" gives us a list, \"xrange\" an iterator.
        octants = xrange(0, n_oct)
        g_octants = [octree.get_global_idx(octant) for octant in octants]
        py_octs = [octree.get_octant(octant) for octant in octants]
        centers = [octree.get_center(octant)[: dimension] for octant in octants]         
        t_foregrounds = self._t_foregrounds
        if (is_background):
            # Current transformation matrix's dictionary.
            c_t_dict = self.get_trans(0)

        # Code hoisting.
        get_corners_from_center = utilities.get_corners_from_center
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons
        get_bound = self._octree.get_bound
        check_neighbour = self.check_neighbour

        for octant in octants:
            d_count, o_count = 0, 0
            g_octant = g_octants[octant]
            py_oct = py_octs[octant]
            # Lambda function.
            g_b = lambda x : get_bound(py_oct, x)
            center  = centers[octant]
            # Check to know if an octant is penalized.
            is_penalized = False
            # Background grid.
            if (is_background):
                is_penalized = True
                threshold = 0.0
                oct_corners = get_corners_from_center(center,
                                                      h)
                for i, corner in enumerate(oct_corners):
                    is_corner_penalized = False
                    corner = apply_persp_trans(dimension, 
                                               corner   , 
                                               c_t_dict ,
                                               logger   ,  
                                               log_file)[: dimension]
                    (is_corner_penalized,
                     n_polygon) = is_point_inside_polygons(corner       ,
                                                           t_foregrounds,
                                                           logger       ,
                                                           log_file     ,
                                                           threshold)
                    if (not is_corner_penalized):
                        is_penalized = False
                        break

            if (is_penalized):
                self._nln[octant] = -1
                key = (n_polygon, # Foreground grid to which the node belongs to
                       g_octant , # Global index (not yet masked)
                       h)         # Node edge's size
                if (self._p_inter):
                    key = key + (-1, -1,)
                # If the octant is covered by the foreground grids, we need
                # to store info of the stencil it belongs to to push on the
                # relative rows of the matrix, the right indices of the octants
                # of the foreground grid owning the penalized one.
                stencil = [-1] * 43
                stencil[0] = g_octant
                stencil[1], stencil[2] = center
                # http://www.laurentluce.com/posts/python-dictionary-implementation/
                # http://effbot.org/zone/python-hash.htm
                self._edl.update({key : stencil})
            else:
                self._nln[octant] = new_oct_count
                new_oct_count += 1
                d_count += 1
            # First boundary face for foreground grids.
            f_b_face = False
            # \"stencil\"'s index.
            s_i = 3
            # Nodes yet seen.
            n_y_s = set()
            # Nodes to not see.
            n_t_n_s = set()

            # Faces' loop.
            for face in xrange(0, nfaces):
                # Not boundary face.
                if (not g_b(face)):
                    neighs, ghosts = ([] for i in range(0, 2))
                    n_y_s.update(face_node[face][0 : 2])
                    (d_count, 
                     o_count, 
                     s_i) = check_neighbour(1                            ,
                                            face                         ,
                                            neighs                       ,
                                            ghosts                       ,
                                            octant                       ,
                                            o_count                      ,
                                            d_count                      ,
                                            s_i                          ,
                                            h                            ,
                                            key if is_penalized else None,
                                            octree                       ,
                                            is_penalized                 ,
                                            is_background                ,
                                            n_polygon if (is_background) \
                                                      else None          )
                else:
                    # Remove (if present) from set \"n_y_s\" the nodes on the
                    # intersection between an edge on the boundary and an edge
                    # not on the boundary.
                    # Boundary nodes.
                    b_ns =  face_node[face][0 : 2]
                    n_t_n_s.update(b_ns)
                    # Adding elements for the octants of the background to use
                    # to interpolate stencil values for boundary conditions of 
                    # the octants of the foreground grid. 
                    if not is_background:
                        #TODO: Check if foreground grid boundary point is or not
                        #      inside the background grid. For the moment every
                        #      case adds +9 at \"o_count\".
                        o_count += 9 # For the neighbours of node.
            # New set with elements in \"n_y_s\" but not in \"n_t_n_s\". 
            n_y_s = n_y_s.difference(n_t_n_s)
            for node in n_t_n_s:
                if not is_background:
                    o_count += 9
            # Nodes' loop.
            for node in n_y_s:
                neighs, ghosts = ([] for i in range(0, 2))
                (d_count, 
                 o_count, 
                 s_i) = check_neighbour(2                            ,
                                        node                         ,
                                        neighs                       ,
                                        ghosts                       ,
                                        octant                       ,
                                        o_count                      ,
                                        d_count                      ,
                                        s_i                          ,
                                        h                            ,
                                        key if is_penalized else None,
                                        octree                       ,
                                        is_penalized                 ,
                                        is_background                ,
                                        n_polygon if (is_background) \
                                                  else None          )
            if (not is_penalized):
                d_nnz.append(d_count)
                o_nnz.append(o_count)
                self._centers_not_penalized.append(center)

        self.new_spread_new_background_numeration(is_background)

        msg = "Created mask"
        self.log_msg(msg   ,
                     "info")

        return (d_nnz, o_nnz)
    # --------------------------------------------------------------------------
    #@profile
    def new_spread_new_background_numeration(self,
                                             is_background):
        to_send = None
        to_receive = self._ngn
        n_oct = self._n_oct
        comm_l = self._comm
        comm_l_s = comm_l.size
        comm_w = self._comm_w
        comm_w_s = comm_w.size
        rank_l = comm_l.Get_rank()
        tot_not_masked_oct = numpy.sum(self._nln != -1)
        tot_masked_oct = n_oct - tot_not_masked_oct
        # Elements not penalized for grid.
        el_n_p_for_grid = numpy.empty(comm_l_s,
                                      dtype = int)
        comm_l.Allgather(tot_not_masked_oct, 
                         el_n_p_for_grid)
        # Counting the number of octants not penalized owned by all the previous
        # grids to know the offset to add at the global numeration of the octree
        # because although it is global, it is global at the inside of each
        # octant, not in the totality of the grids.
        oct_offset = 0
        for i in xrange(0, rank_l):
            oct_offset += el_n_p_for_grid[i]
        # Adding the offset at the new local numeration.
        self._nln[self._nln >= 0] += oct_offset
        # Send counts. How many element have to be sent by each process.
        self._s_counts = numpy.empty(comm_w_s,
                                     dtype = numpy.int64)
        one_el = numpy.empty(1, 
                             dtype = numpy.int64)
        one_el[0] = self._nln.size
        if (not is_background):
            one_el[0] = 0
            to_send = numpy.zeros(0, dtype = numpy.int64)

        comm_w.Allgather(one_el, 
                         [self._s_counts, 1, MPI.INT64_T])
        displs = numpy.empty(comm_w_s, dtype = numpy.int64)
        # Local displacement.
        l_displ = numpy.zeros(1, dtype = numpy.int64)
        offset = 0
        if (is_background):
            for i in xrange(0, rank_l):
                offset += self._s_counts[i]
            l_displ[0] = offset
            to_send = self._nln

        comm_w.Allgather(l_displ,
                         [displs, 1, MPI.INT64_T])

        comm_w.Allgatherv(to_send,
                          [to_receive, self._s_counts, displs, MPI.INT64_T])

        msg = "Spread new global background masked numeration"
        self.log_msg(msg   ,
                     "info")
        

    # --------------------------------------------------------------------------
    # The new masked global numeration for the octants of the background grid 
    # has to be spread to the other meshes.
    def spread_new_background_numeration(self,
                                         is_background):
        n_oct = self._n_oct
        comm_l = self._comm
        comm_l_s = comm_l.size
        comm_w = self._comm_w
        rank_l = comm_l.Get_rank()
        tot_not_masked_oct = numpy.sum(self._nln != -1)
        tot_masked_oct = n_oct - tot_not_masked_oct
        # Elements not penalized for grid.
        el_n_p_for_grid = numpy.empty(comm_l_s,
                                      dtype = int)
        comm_l.Allgather(tot_not_masked_oct, 
                         el_n_p_for_grid)
        # Counting the number of octants not penalized owned by all the previous
        # grids to know the offset to add at the global numeration of the octree
        # because although it is global, it is global at the inside of each
        # octant, not in the totality of the grids.
        oct_offset = 0
        for i in xrange(0, comm_l_s):
            if i < rank_l:
                oct_offset += el_n_p_for_grid[i]
        # Adding the offset at the new local numeration.
        self._nln[self._nln >= 0] += oct_offset
        
        if is_background:
            # Send counts. How many element have to be sent by each process.
            self._s_counts = numpy.empty(comm_l_s,
                                         dtype = numpy.int64)
            one_el = numpy.empty(1, 
                                 dtype = numpy.int64)
            one_el[0] = self._nln.size
            comm_l.Allgather(one_el, 
                             [self._s_counts, 1, MPI.INT64_T])
            displs = [0] * comm_l_s
            offset = 0
            for i in xrange(1, comm_l_s):
                offset += self._s_counts[i-1]
                displs[i] = offset
            comm_l.Gatherv(self._nln                                       ,
                           [self._ngn, self._s_counts, displs, MPI.INT64_T],
                           root = 0)
        # Broadcasting the vector containing the new global numeration of the
        # background grid \"self._ngn\" to all processes of the world 
        # communicator.
        N_oct_bg_g = self._oct_f_g[0]
        comm_w.Bcast([self._ngn, N_oct_bg_g, MPI.INT64_T],
                     root = 0)

        msg = "Spread new global background masked numeration"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Initialize diagonal matrices of the block matrix.
    def init_mat(self              ,
                 (e_d_nnz, e_o_nnz)):
        """Method which initialize the diagonal parts of the monolithic matrix 
           of the system.
           
           Arguments:
               o_n_oct (int) : number of octants overlapped."""

	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        f_bound = self._f_bound
        grid = self._proc_g
        comm_w = self._comm_w
        rank_w = self._rank_w
        octree = self._octree
        tot_oct = self._tot_oct
        is_background = False if (grid) else True
        # Range deplacement.
        h = self._h
        h_half = 0.5 * h
        h2 = h * h
        h2_inv = 1.0 / h2
        oct_offset = 0
        t_foregrounds = self._t_foregrounds
        for i in xrange(0, grid):
            oct_offset += self._oct_f_g[i]

        (d_nnz, o_nnz) = (e_d_nnz, e_o_nnz)
        n_oct = self._n_oct
        nfaces = octree.get_n_faces()
        face_node = octree.get_face_node()
        #TODO: check function. 
        sizes = self.find_sizes()

        self._b_mat = PETSc.Mat().createAIJ(size = (sizes, sizes),
                                            nnz = (d_nnz, o_nnz) ,
                                            comm = comm_w)
        # If an element is being allocated in a place not preallocate, then 
        # the program will stop.
        self._b_mat.setOption(self._b_mat.Option.NEW_NONZERO_ALLOCATION_ERR, 
                              True)
        #TODO: check function. 
        o_ranges = self.get_ranges()

        dimension = self._dim
        # Current transformation matrix's dictionary.
        c_t_dict = self.get_trans(grid)
        # Current transformation adjoint matrix's dictionary.
        c_t_a_dict = self.get_trans_adj(grid) 
        
        # Code hoisting.
        mask_octant = self.mask_octant
        get_octant = octree.get_octant
        get_center = octree.get_center
        get_corners_from_center = utilities.get_corners_from_center
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons
        metric_coefficients = utilities.metric_coefficients
        square = numpy.square
        check_neighbour = self.check_neighbour
        get_bound = octree.get_bound
        # Lambda function.
        g_c_f_c = lambda x : get_corners_from_center(x, h)

        for octant in xrange(0, n_oct):
            indices, values = ([] for i in range(0, 2)) # Indices/values
            neighs, ghosts = ([] for i in range(0, 2))
            g_octant = o_ranges[0] + octant
            # Masked global octant.
            m_g_octant = mask_octant(g_octant)
            py_oct = get_octant(octant)
            center = get_center(octant)[: dimension]
            # Lambda function.
            g_b = lambda x : get_bound(py_oct, x)
            # Check to know if an octant on the background is penalized.
            is_penalized = False
            # Background grid.
            if (is_background):
                is_penalized = True
                threshold = 0.0
                oct_corners = g_c_f_c(center)
                for i, corner in enumerate(oct_corners):
                    is_corner_penalized = False
                    corner = apply_persp_trans(dimension, 
                                               corner   , 
                                               c_t_dict ,
                                               logger   ,  
                                               log_file)[: dimension]
                    (is_corner_penalized,
                     n_polygon) = is_point_inside_polygons(corner       ,
                                                           t_foregrounds,
                                                           logger       ,
                                                           log_file     ,
                                                           threshold)
                    if (not is_corner_penalized):
                        is_penalized = False
                        break

            if (not is_penalized):
                indices.append(m_g_octant)
                t_center = apply_persp_trans(dimension, 
                                             center   , 
                                             c_t_dict ,
                                             logger   ,  
                                             log_file)[: dimension]
                i_metric_coefficients = metric_coefficients(dimension,
                                                            [t_center],
                                                            c_t_a_dict,
                                                            logger,
                                                            log_file)
                A00 = i_metric_coefficients[0] 
                A10 = i_metric_coefficients[1]
                A01 = i_metric_coefficients[2]
                A11 = i_metric_coefficients[3]
                ds2_epsilon_x = i_metric_coefficients[4]
                ds2_epsilon_y = i_metric_coefficients[5]
                ds2_nu_x = i_metric_coefficients[6]
                ds2_nu_y = i_metric_coefficients[7]
                A002 = square(A00)
                A102 = square(A10)
                A012 = square(A01)
                A112 = square(A11)
                ds2_epsilon_x = i_metric_coefficients[4]
                ds2_epsilon_y = i_metric_coefficients[5]
                ds2_nu_x = i_metric_coefficients[6]
                ds2_nu_y = i_metric_coefficients[7]
                t_m = (-2.0) * ((A002 + A102) + (A012 + A112))
                value_to_append = t_m * h2_inv
                values.append(value_to_append)
                # Nodes yet seen.
                n_y_s = set()
                # Nodes to not see.
                n_t_n_s = set()
                for face in xrange(0, nfaces):
                    if (not g_b(face)):
                        n_y_s.update(face_node[face][0 : 2])

                        (m_index       , 
                         is_n_penalized,
                         n_center) = check_neighbour(1                         ,
                                                     face                      ,
                                                     neighs                    ,
                                                     ghosts                    ,
                                                     octant                    ,
                                                     oct_offset                ,
                                                     o_ranges                  ,
                                                     0                         ,
                                                     h                         ,
                                                     None                      ,
                                                     octree                    ,
                                                     is_penalized              ,
                                                     is_background             ,
                                                     n_polygon if is_background\
                                                               else None       ,
                                                     yet_masked = True)
                        if not is_n_penalized:
                            indices.append(m_index)
                            # Temporary multiplier.
                            t_m = (A002 + A102) if (face < 2) else \
                                  (A012 + A112)
                            # Temporary multiplier 02.
                            t_m02 = (ds2_epsilon_x + ds2_epsilon_y) if \
                                    (face < 2) else (ds2_nu_x + ds2_nu_y)
                            t_m02 *= h_half
                            t_m += t_m02 if ((face % 2) == 1) else \
                                   (-1.0 * t_m02)   
                            value_to_append = t_m * h2_inv
                            values.append(value_to_append)
                    else:
                        # Removing nodes being on the boundaries. They will not
                        # be counted in the stencil.
                        b_ns = face_node[face][0 : 2]
                        n_t_n_s.update(b_ns)
                # New set with elements in \"n_y_s\" but not in \"n_t_n_s\". 
                n_y_s = n_y_s.difference(n_t_n_s)
                for node in n_y_s:
                    (m_index       , 
                     is_n_penalized,
                     n_center) = check_neighbour(2                         ,
                                                 node                      ,
                                                 neighs                    ,
                                                 ghosts                    ,
                                                 octant                    ,
                                                 oct_offset                ,
                                                 o_ranges                  ,
                                                 0                         ,
                                                 h                         ,
                                                 None                      ,
                                                 octree                    ,
                                                 is_penalized              ,
                                                 is_background             ,
                                                 n_polygon if is_background\
                                                           else None       ,
                                                 yet_masked = True)
                    if not is_n_penalized:
                        indices.append(m_index)
                        t_m = (A00 * A01) + (A10 * A11)
                        t_m *= 0.5 if ((node == 0) or (node == 3)) \
                                   else -0.5
                        value_to_append = t_m * h2_inv
                        values.append(value_to_append)

                self._b_mat.setValues(m_g_octant, # Row
                                      indices   , # Columns
                                      values)     # Values to be inserted
    
        # We have inserted argument \"assebly\" equal to 
        # \"PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY\" because the final assembly
        # will be done after inserting the prolongation and restriction blocks.
        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY)
        
        msg = "Initialized diagonal parts of the monolithic  matrix"
        extra_msg = "with sizes \"" + str(self._b_mat.getSizes()) + \
                    "\" and type \"" + str(self._b_mat.getType()) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Assembly PETSc's structures, like \"self._b_mat\" or \"self._rhs\".
    def assembly_petsc_struct(self       ,
                              struct_type,
                              assembly_type = None):
        """Function which inglobe \"assemblyBegin()\" and \"assemblyEnd()\" 
           \"PETSc\" function to avoid problem of calling other functions
           between them.
        
           Arguments:
                struct_type (string) : what problem's structure we want to
                                       assembly.
                assembly_type (PETSc.Mat.AssemblyType) : type of assembly; it
                                                         can be \"FINAL\" or
                                                         \"FLUSH\"."""

        if struct_type == "matrix":
            self._b_mat.assemblyBegin(assembly = assembly_type)
            self._b_mat.assemblyEnd(assembly = assembly_type)
        elif struct_type == "rhs":
            self._rhs.assemblyBegin()
            self._rhs.assemblyEnd()
        else:
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " PETSc struct " + str(struct_type) +\
                        "not recognized."
            self.log_msg(msg    , 
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1) 
        
        msg = "Assembled PETSc structure " + str(struct_type)
        extra_msg = "with assembly type " + str(assembly_type)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Finds local and global sizes used to initialize \"PETSc \" structures.
    def find_sizes(self):
	"""Method which find right sizes for \"PETSc\" structures.
 
	   Returns:
	        sizes (tuple) : sizes for \"PETSc\" data structure."""

        grid = self._proc_g
        n_oct = self._n_oct
        rank_l = self._rank
        not_masked_oct_bg_g = numpy.size(self._ngn[self._ngn != -1])
        self._masked_oct_bg_g = self._ngn.size - not_masked_oct_bg_g
        tot_oct = self._tot_oct - self._masked_oct_bg_g 
        if not grid:
            # Not masked local octant background grid.
            not_masked_l_oct_bg_g = numpy.size(self._nln[self._nln != -1])
        sizes = (n_oct if grid else not_masked_l_oct_bg_g, 
                 tot_oct)

        msg = "Found sizes for PETSc structure"
        extra_msg = "with sizes " + str(sizes)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        return sizes
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Interpolating the solution is necessary to be able to use the \"vtm\"
    # typology of files. For the moment, the value interpolated is substituted
    # by the fixed value \"0.0\".
    def reset_partially_solution(self):
        """Function which creates a \"new\" solution array, pushing in the
           octants covered by foreground meshes the values interpolated from the
           neighbours around them.

           Returns:
                res_sol (PETSc.Vec) : the partially resetted solution."""

        grid = self._proc_g
        octree = self._octree
        o_ranges = self.get_ranges()
        n_oct = self._n_oct
        tot_oct = self._tot_oct
        # Upper bound octree's id contained.
        up_id_octree = o_ranges[0] + n_oct
        # Octree's ids contained.
        ids_octree_contained = xrange(o_ranges[0], 
                                      up_id_octree)
        # Resetted solution.
        res_sol = self.init_array("resetted partially solution",
                                  petsc_size = False)

        for i in ids_octree_contained:
            sol_index = self.mask_octant(i)
            if (sol_index != -1):
                sol_value = self._sol.getValue(sol_index)
                res_sol.setValue(i, sol_value)

        res_sol.assemblyBegin()
        res_sol.assemblyEnd()
        
        msg = "Resetted partially solution"
        self.log_msg(msg   ,
                     "info")
    
        return res_sol
    # --------------------------------------------------------------------------
   
    # --------------------------------------------------------------------------
    # Initializes a \"PTESc\" array, being made of zeros or values passed by.
    def init_array(self             ,
                   # Array name.
                   a_name = ""      ,
                   petsc_size = True,
                   array = None):
	"""Method which initializes an array or with zeros or with a 
	   \"numpy.ndarray\" passed as parameter.

	   Arguments:
		a_name (string) : name of the array to initialize, being written
				  into the log. Default value is \"\".
                petsc_size (boolean): if \"True\", it impose the use of the 
                                      sizes used by \"PETSc\". Otherwise, the
                                      local and global numbers of octants from
                                      the octree are used. 
		array (numpy.ndarray) : possible array to use to initialize the
					returned array. Default value is 
					\"None\".

	   Returns:
		a PETSc array."""
	
        if not petsc_size:
            n_oct = self._n_oct
            tot_oct = self._tot_oct
            sizes = (n_oct, tot_oct)
        else: 
            sizes = self.find_sizes()
        # Temporary array.
        t_array = PETSc.Vec().createMPI(size = sizes,
                                        comm = self._comm_w)
        t_array.setUp()

        if array is None:
            t_array.set(0)
        else:
            try:
                assert isinstance(array, numpy.ndarray)
                # Temporary PETSc vector.
                t_petsc = PETSc.Vec().createWithArray(array       ,
                                                      size = sizes,
                                                      comm = self._comm_w)
                t_petsc.copy(t_array)
            except AssertionError:
                msg = "\"MPI Abort\" called during array's initialization"
                extra_msg = "Parameter \"array\" not an instance of " + \
                            "\"numpy.ndarray\"."
                self.log_msg(msg    ,
                             "error",
                             extra_msg)
                self._comm_w.Abort(1)
        msg = "Initialized \"" + str(a_name) + "\""
        self.log_msg(msg,
                     "info")
        return t_array
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initializes \"rhs\".
    def init_rhs(self, 
                 numpy_array):
	"""Method which intializes the right hand side.
            
           Arguments:
                numpy_array (numpy.array) : array to use to initialize with it
                                            the \"rhs\"."""

        self._rhs = self.init_array("right hand side",
                                    True             ,
                                    numpy_array)
        
        msg = "Initialized \"rhs\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initializes \"sol\".
    def init_sol(self):
        """Method which initializes the solution."""

        self._sol = self.init_array("solution",
                                    True)
        
        msg = "Initialized \"rhs\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Solving...
    def solve(self):
        """Method which solves the system."""
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        pc = PETSc.PC()
        ksp.create(self._comm_w)
        ksp.setOperators(self._b_mat,
                         None)

        # Setting tolerances.
	# 1.0e-06 with level 9,9 is still ok for convergence
        tol = 1.e-06
        ksp.setTolerances(rtol = tol            , 
                          atol = tol            , 
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
        ksp.setType("gmres")
        ksp.setFromOptions()
        # Solve the system.
        #self._b_mat.view()
        ksp.solve(self._rhs, 
                  self._sol)
        # How many iterations are done.
        it_number = ksp.getIterationNumber()
        #print(ksp.getConvergedReason())
        #self._b_mat.view()

        msg = "Evaluated solution"
        extra_msg = "Using \"" + str(it_number) + "\" iterations."
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initialize exchanged structures.	
    def init_e_structures(self):
	"""Method which initializes structures used to exchange data between
	   different grids."""

        grid = self._proc_g
        is_background = True
        if grid:
            is_background = False
        n_oct = self._n_oct
        N_oct_bg_g = self._oct_f_g[0]
        # The \"self._edl\" will contains local data to be exchanged between
	# grids of different levels.
	# Exchanged data local.
        self._edl = {} 
        # The \"self._edg_c\" will contains the count of data between
        # grids of different levels.
	# Exchanged data global count, for the moment initialize with size one.
        self._edg_c = numpy.empty(self._comm_w.Get_size() - \
                                  self._comm.Get_size(),
                                  dtype = numpy.int64)
        # New local numeration. 
        self._nln = numpy.empty(n_oct,
                                dtype = numpy.int64)
        # New global numeration.
        self._ngn = numpy.empty(N_oct_bg_g,
                                dtype = numpy.int64)
        self._centers_not_penalized = []

        # Numpy edl.
        self._n_edl = None
        # Numpy edg. The \"self._n_edg\" will contains the excahnged data 
        # between grids of different levels.
        self._n_edg = None
        if not is_background:
            self._d_type_s = numpy.dtype('(1, 5)f8, (1, 43)f8') if self._p_inter\
                             else numpy.dtype('(1, 5)f8, (1, 2)f8')
            blocks_length_s = [5, 43] if self._p_inter else [5, 2]
            blocks_displacement_s = [0, 40]
            mpi_datatypes = [MPI.DOUBLE,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 5)f8, (1, 43)f8') if self._p_inter\
                             else numpy.dtype('(1, 3)f8, (1, 43)f8')
            blocks_length_r = [5, 43] if self._p_inter else [3, 43]
            blocks_displacement_r = [0, 40] if self._p_inter else [0, 24]
        else:
            self._d_type_s = numpy.dtype('(1, 5)f8, (1, 43)f8') if self._p_inter\
                             else numpy.dtype('(1, 3)f8, (1, 43)f8')
            blocks_length_s = [5, 43] if self._p_inter else [3, 43]
            blocks_displacement_s = [0, 40] if self._p_inter else [0, 24]
            mpi_datatypes = [MPI.DOUBLE,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 5)f8, (1, 43)f8') if self._p_inter\
                             else numpy.dtype('(1, 5)f8, (1,2)f8')
            blocks_length_r = [5, 43] if self._p_inter else [5, 2]
            blocks_displacement_r = [0, 40]
        # MPI data type to send.
        self._mpi_d_t_s = MPI.Datatype.Create_struct(blocks_length_s      ,
                                                     blocks_displacement_s,
                                                     # Do not forget 
                                                     #\".Commit()\".
                                                     mpi_datatypes).Commit()
        # MPI data type to receive.
        self._mpi_d_t_r = MPI.Datatype.Create_struct(blocks_length_r      ,
                                                     blocks_displacement_r,
                                                     mpi_datatypes).Commit()

        msg = "Initialized exchanged structures"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Creating the restriction and prolongation blocks inside the monolithic 
    # matrix of the system.
    def update_values(self                , 
                      intercomm_dictionary):
        """Method wich update the system matrix.

           Arguments:
                intercomm_dictionary (python dict) : contains the
                                                     intercommunicators."""

        log_file = self.logger.handlers[0].baseFilename
        grid = self._proc_g
        n_oct = self._n_oct
        comm_w = self._comm_w
        comm_l = self._comm
        rank_w = self._rank_w
        rank_l = self._rank
        is_background = True
        if grid:
            is_background = False
        o_ranges = self.get_ranges()
        # Upper bound octree's id contained.
        up_id_octree = o_ranges[0] + n_oct
        # Octree's ids contained.
        ids_octree_contained = (o_ranges[0], up_id_octree)
        self._n_edl = numpy.array(self._edl.items(), 
                                  dtype = self._d_type_s)
        mpi_requests = []
        one_el = numpy.empty(1, 
                             dtype = numpy.int64)
        one_el[0] = len(self._edl)
        displ = 0

        if (not self._p_inter) and (grid):
            r_g_s = intercomm_dictionary.values()[0].Get_remote_size()
            self._edg_c = numpy.resize(self._edg_c,
                                       r_g_s)

        for key, intercomm in intercomm_dictionary.items():
            req = intercomm.Iallgather(one_el,
                                       [self._edg_c, 1, displ, MPI.INT64_T])
            mpi_requests.append(req)

            r_g_s = intercomm.Get_remote_size()
            displ += r_g_s

        for i, mpi_request in enumerate(mpi_requests):
            status = MPI.Status()
            mpi_request.Wait(status)

        t_length = 0
        for index, size_edl in enumerate(self._edg_c):
            t_length += size_edl

        self._n_edg = numpy.zeros(t_length, 
                                  dtype = self._d_type_r)

        displs = [0] * len(self._edg_c)
        offset = 0
        for i in range(1, len(self._edg_c)):
            offset += self._edg_c[i-1]
            displs[i] = offset

        # \"self._n_edg\" position.
        n_edg_p = 0
        mpi_requests = []
        for key, intercomm in intercomm_dictionary.items():
            # Remote group size.
            r_g_s = intercomm.Get_remote_size()
            i = n_edg_p
            j = i + r_g_s 
            req = intercomm.Iallgatherv([self._n_edl, self._mpi_d_t_s],
                                        [self._n_edg       , 
                                         self._edg_c[i : j],
                                         displs[i : j]     , 
                                         self._mpi_d_t_r])
            n_edg_p += r_g_s
            mpi_requests.append(req)
        
        for i, mpi_request in enumerate(mpi_requests):
            status = MPI.Status()
            mpi_request.Wait(status)
        if not is_background:
            self.update_fg_grids(o_ranges,
                                 ids_octree_contained)
        else:
            self.update_bg_grids(o_ranges,
                                 ids_octree_contained)
        
        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)

        self.assembly_petsc_struct("rhs")
        
        msg = "Updated monolithic matrix"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Sub_method of \"update_values\".
    def update_fg_grids(self                ,
                        o_ranges            ,
                        ids_octree_contained):
        """Method which update the non diagonal blocks relative to the 
           foreground grids.

           Arguments:
                o_ranges (tuple) : the initial and final octants managed by the
                                   current process.
                ids_octree_contained (list) : list of the indices of the octants
                                              contained in the current process."""

	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger 
        octree = self._octree
        comm_l = self._comm
        dimension = self._dim
        proc_grid = self._proc_g
        # Background transformation matrix adjoint's dictionary.
        b_t_adj_dict = self.get_trans_adj(0)
        # Background transformation matrix dictionary.
        b_t_dict = self.get_trans(0)
        # Current transformation matrix adjoint's dictionary.
        c_t_adj_dict = self.get_trans_adj(proc_grid)

        #start = time.time()
        list_edg = list(self._n_edg)
        # Length list edg.
        l_l_edg = len(list_edg)
        p_inter = self._p_inter
        if (p_inter):
            list_edg = [list_edg[i] for i in xrange(0, l_l_edg) if
                        int(list_edg[i][0].item(0)) == 0]
        # Length list edg (new).
        l_l_edg = len(list_edg)
        # Length key.
        l_k = list_edg[0][0].size
        # Length stencil.
        l_s = list_edg[0][1].size
        keys = numpy.array([list_edg[i][0] for i in 
                            range(0, l_l_edg)]).reshape(l_l_edg, l_k)
        h2s = keys[:, 2] * keys[:, 2]
        stencils = numpy.array([list_edg[i][1] for i in 
                                # TODO: 12 or 16 instead of 9 for grid not
                                # perfectly superposed??
                                range(0, l_l_edg)]).reshape(l_l_edg, l_s)
        centers = [(stencils[i][1], stencils[i][2], 0) for i in range(0, l_l_edg)]
        n_centers = len(centers)
        t_centers = [None] * n_centers

        # Code hoisting.
        apply_persp_trans = utilities.apply_persp_trans
        apply_persp_trans_inv = utilities.apply_persp_trans_inv
        find_right_neighbours = self.find_right_neighbours
        least_squares = utilities.least_squares
        metric_coefficients = utilities.metric_coefficients
        apply_rest_prol_ops = self.apply_rest_prol_ops

        # Lambda function.
        f_r_n = lambda x : find_right_neighbours(x, o_ranges[0])
        
        for i in xrange(0, n_centers):
                centers[i] = apply_persp_trans(dimension , 
                                               centers[i], 
                                               b_t_dict  ,
                                               logger    ,  
                                               log_file)[: dimension]
                t_centers[i] = centers[i]
                centers[i] = apply_persp_trans_inv(dimension   ,
                                                   centers[i]  ,
                                                   c_t_adj_dict,
                                                   logger      ,
                                                   log_file)[: dimension]
        # Vectorized functions are just syntactic sugar:
        # http://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array
        # http://stackoverflow.com/questions/8079061/function-application-over-numpys-matrix-row-column
        # http://stackoverflow.com/questions/6824122/mapping-a-numpy-array-in-place
        # http://stackoverflow.com/questions/9792925/how-to-speed-up-enumerate-for-numpy-array-how-to-enumerate-over-numpy-array-ef
        local_idxs = numpy.array([octree.get_point_owner_idx((center[0],
                                                              center[1],
                                                              center[2] if    \
                                                              (dimension == 3)\
                                                              else 0)) for 
                                  center in centers])
        global_idxs = local_idxs + o_ranges[0]
        # \"numpy.where\" returns indices of the elements which satisfy the
        # conditions.
        idxs = numpy.where(numpy.logical_and((global_idxs >= 
                                              ids_octree_contained[0]),
                                             (global_idxs <= 
                                              ids_octree_contained[1])))
        # \"idxs[0]\" because is a numpy array, so to select the array we have
        # to use the index notation.
        for idx in idxs[0]:
            neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
            (neigh_centers, 
             neigh_indices)  = f_r_n(local_idxs[idx])
            coeffs = least_squares(neigh_centers,
                                   centers[idx])
            h2 = h2s[idx]
            h = numpy.sqrt(h2)
            h2_inv = 1.0 / h2
            h_half = 0.5 * h

            # Checkout how the \"stencil\" is created in the function
            # \"create_mask\".
            for i in xrange(6, len(stencils[idx]), 5):
                b_codim = int(stencils[idx][i])
                index = int(stencils[idx][i + 1])
                x_s = t_centers[idx][0]
                y_s = t_centers[idx][1]

                row_index = int(stencils[idx][i - 3])
                n_center = [stencils[idx][i - 2], stencils[idx][i - 1]]
                t_n_center = apply_persp_trans(dimension, 
                                               n_center , 
                                               b_t_dict ,
                                               logger   ,  
                                               log_file)[: dimension]

                i_metric_coefficients = metric_coefficients(dimension   ,
                                                            [t_n_center],
                                                            b_t_adj_dict,
                                                            logger      ,
                                                            log_file)
                A00 = i_metric_coefficients[0] 
                A10 = i_metric_coefficients[1]
                A01 = i_metric_coefficients[2]
                A11 = i_metric_coefficients[3]
                ds2_epsilon_x = i_metric_coefficients[4]
                ds2_epsilon_y = i_metric_coefficients[5]
                ds2_nu_x = i_metric_coefficients[6]
                ds2_nu_y = i_metric_coefficients[7]
                A002 = numpy.square(A00)
                A102 = numpy.square(A10)
                A012 = numpy.square(A01)
                A112 = numpy.square(A11)
                if (b_codim == 1):
                    # Temporary multiplier.
                    t_m = (A002 + A102) if (index < 2) else (A012 + A112)
                    # Temporary multiplier 02.
                    t_m02 = (ds2_epsilon_x + ds2_epsilon_y) if \
                            (index < 2) else (ds2_nu_x + ds2_nu_y)
                    t_m02 *= h_half
                    # IMPORTANT
                    # Note that here the signs MUST be opposite respect to the same part
                    # into "set_b_c" and "update_bg_grids" functions, because they have to
                    # be considered as stencil of the neighbours, and so with opposite face.
                    t_m += t_m02 if ((index % 2) == 0) else \
                           (-1.0 * t_m02)   
                    value_to_multiply = h2_inv * t_m
                elif (b_codim == 2):
                    t_m = (A00 * A01) + (A10 * A11)
                    t_m *= 0.5 if ((index == 0) or (index == 3)) \
                               else -0.5
                    value_to_multiply = h2_inv * t_m
                elif (b_codim ==  -1):
                    break
                new_coeffs = [coeff * value_to_multiply for coeff in \
                              coeffs]
                apply_rest_prol_ops(row_index     ,
                                    neigh_indices ,
                                    new_coeffs,
                                    neigh_centers)

        #end = time.time()
        #print("fg update " + str(end - start))

        msg = "Updated prolongation blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Sub_method of \"update_values\".
    def update_bg_grids(self                ,
                        o_ranges            ,
                        ids_octree_contained):
        """Method which update the non diagonal blocks relative to the 
           backgrounds grids.

           Arguments:
                o_ranges (tuple) : the initial and final octants managed by the
                                   current process.
                ids_octree_contained (list) : list of the indices of the octants
                                              contained in the current process."""

	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger 
        octree = self._octree
        comm_l = self._comm
        time_rest_prol = 0
        dimension = self._dim
        proc_grid = self._proc_g
        # Current transformation matrix adjoint's dictionary.
        c_t_adj_dict = self.get_trans_adj(proc_grid)

        #start = time.time()
        list_edg = list(self._n_edg)
        # Length list edg.
        l_l_edg = len(list_edg)
        # Length key.
        l_k = list_edg[0][0].size
        # Length stencil.
        l_s = list_edg[0][1].size
        keys = numpy.array([list_edg[i][0] for i in 
                            xrange(0, l_l_edg)]).reshape(l_l_edg, l_k)
        h2s = keys[:, 4] * keys[:, 4]
        # Outside centers.
        o_centers = numpy.array([list_edg[i][1][0][: dimension] for i in 
                                 range(0, l_l_edg)]).reshape(l_l_edg, dimension)
        # Inside centers.
        i_centers = numpy.array([list_edg[i][1][0][dimension : (dimension * 2)]\
                                 for i in range(0, l_l_edg)]).reshape(l_l_edg, 
                                                                      dimension)
        # Number of outside centers.
        n_o_centers = o_centers.shape[0]
        t_o_centers = [None] * n_o_centers
        
        # Code hoisting.
        apply_persp_trans = utilities.apply_persp_trans
        apply_persp_trans_inv = utilities.apply_persp_trans_inv
        find_right_neighbours = self.find_right_neighbours
        least_squares = utilities.least_squares
        metric_coefficients = utilities.metric_coefficients
        apply_rest_prol_ops = self.apply_rest_prol_ops
        get_trans_adj = self.get_trans_adj
        get_trans = self.get_trans

        for i in xrange(0, n_o_centers):
            f_t_dict = self.get_trans(int(keys[i][0]))
            t_o_centers[i] = apply_persp_trans(dimension   , 
                                               o_centers[i], 
                                               f_t_dict    ,
                                               logger      ,  
                                               log_file)
            t_o_centers[i] = apply_persp_trans_inv(dimension     , 
                                                   t_o_centers[i], 
                                                   c_t_adj_dict  ,
                                                   logger        ,  
                                                   log_file)
        #TODO: understand why here we need to pass \"center[0:2]\" to the 
        # function \"get_point_owner_dx\", while in the previous version of
        # PABLitO we passed all the array \"center\". I think that it is due to
        # the change of type of the input arguments from \"dvector\" to 
        # \"darray\".
        local_idxs = numpy.array([octree.get_point_owner_idx((center[0], 
                                                              center[1],
                                                              center[2] if    \
                                                              (dimension == 3)\
                                                              else 0)) for    \
                                  center in t_o_centers])
        global_idxs = local_idxs + o_ranges[0]
        idxs = numpy.where(numpy.logical_and((global_idxs >= 
                                              ids_octree_contained[0]),
                                             (global_idxs <= 
                                              ids_octree_contained[1])))
        #print(str(self._comm_w.Get_rank()) + " " + str(len(idxs[0])))
        for idx in idxs[0]:
            # Foreground transformation matrix adjoint's dictionary.
            f_t_adj_dict = get_trans_adj(int(keys[idx][0]))
            f_t_dict = get_trans(int(keys[idx][0]))
            t_i_center =  apply_persp_trans(dimension     , 
                                            i_centers[idx], 
                                            f_t_dict      ,
                                            logger        ,  
                                            log_file)[: dimension]

            i_metric_coefficients = metric_coefficients(dimension,
                                                        [t_i_center],
                                                        f_t_adj_dict,
                                                        logger,
                                                        log_file)
            
            neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
            # New neighbour indices.
            n_n_i = []
            (neigh_centers, 
             neigh_indices)  = find_right_neighbours(local_idxs[idx],
                                                     o_ranges[0]    ,
                                                     True)
            coeffs = least_squares(neigh_centers,
                                   t_o_centers[idx])
            
            n_n_i = neigh_indices

            b_codim = int(keys[idx][3])
            index = int(keys[idx][2])

            h2 = h2s[idx]
            h = numpy.sqrt(h2)
            h2_inv = 1.0 / h2
            h_half = 0.5 * h

            A00 = i_metric_coefficients[0] 
            A10 = i_metric_coefficients[1]
            A01 = i_metric_coefficients[2]
            A11 = i_metric_coefficients[3]
            ds2_epsilon_x = i_metric_coefficients[4]
            ds2_epsilon_y = i_metric_coefficients[5]
            ds2_nu_x = i_metric_coefficients[6]
            ds2_nu_y = i_metric_coefficients[7]
            A002 = numpy.square(A00)
            A102 = numpy.square(A10)
            A012 = numpy.square(A01)
            A112 = numpy.square(A11)
            if (b_codim == 1):
                # Temporary multiplier.
                t_m = (A002 + A102) if (index < 2) else (A012 + A112)
                # Temporary multiplier 02.
                t_m02 = (ds2_epsilon_x + ds2_epsilon_y) if \
                        (index < 2) else (ds2_nu_x + ds2_nu_y)
                t_m02 *= h_half
                t_m += t_m02 if ((index % 2) == 1) else \
                       (-1.0 * t_m02)   

                value_to_multiply = h2_inv * t_m
            else:
                t_m = (A00 * A01) + (A10 * A11)
                t_m *= 0.5 if ((index == 0) or (index == 3)) \
                           else -0.5
                value_to_multiply = h2_inv * t_m

            coeffs = [coeff * value_to_multiply for coeff in coeffs]
            apply_rest_prol_ops(int(keys[idx][1]),
                                n_n_i            ,
                                coeffs           ,
                                neigh_centers)
        #end = time.time()
        #print("bg update " + str(end - start))

        msg = "Updated restriction blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # TODO: Find a more generic algortihm: try least squares.
    # --------------------------------------------------------------------------
    # Returns the right neighbours for an octant, being them of edges or nodes.
    def find_right_neighbours(self                 , 
                              current_octant       ,
                              start_octant         ,
                              is_background = False):
        """Method which compute the right 4 neighbours for the octant 
           \"current_octant\", considering first the label \"location\" to
           indicate in what directions go to choose the neighborhood.

           Arguments:
                current_octant (int) : local index of the current octant.
                start_octant (int) : global index of the first contained octant
                                     in the process.
                is_background (bool) : indicates if we are or not on the 
                                       background grid. On this choice depends
                                       how the indices of the neighbours will
                                       be evaluated.

           Returns:
                (centers, indices) (tuple of lists) : tuple containing the lists
                                                      of centers and indices of
                                                      the neighbours."""

        py_oct = self._octree.get_octant(current_octant)
        centers = []
        indices = []
        grid = self._proc_g
        dimension = self._dim
        octree = self._octree
        nfaces = octree.get_n_faces()
        nnodes = octree.get_n_nodes()
        c_t_dict = self.get_trans(grid)
        t_background = self._t_background
	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        # Ghosts' deplacement.
        g_d = 0
        for i in xrange(0, grid):
            g_d = g_d + self._oct_f_g[i]
        # Current center.
        c_c = self._octree.get_center(current_octant)[: dimension]

        centers.append(c_c)
        index = current_octant
        m_index = self.mask_octant(index + start_octant)
        indices.append(m_index)

        neighs, ghosts = ([] for i in range(0, 2))

        #Code hoisting.
        find_neighbours = octree.find_neighbours
        mask_octant = self.mask_octant
        get_center = octree.get_center
        get_ghost_global_idx = octree.get_ghost_global_idx
        get_ghost_octant = octree.get_ghost_octant
        neighbour_centers = self.neighbour_centers
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygon = utilities.is_point_inside_polygon
        # Lambda function.
        f_n = lambda x,y : find_neighbours(current_octant,
                                           x             ,
                                           y             ,
                                           neighs        ,
                                           ghosts)
        for i in xrange(0, nfaces + nnodes):
            # Codimension.
            codim = 1 if (i <= 3) else 2
            # Index of current face or node.
            face_node = i if (i <= 3) else (i - 4)
            
            (neighs, ghosts) = f_n(face_node, codim)
            # Check if it is really a neighbour of edge or node. If not,
            # it means that we are near the boundary if we are on the
            # background, in an outside area if we are on the foreground, 
            # and so...
            if (not not neighs):
                # Neighbour is into the same process, so is local.
                if (not ghosts[0]):
                    by_octant = False
                    index = neighs[0]
                    m_index = mask_octant(index + start_octant)
                    py_ghost_oct = index
                else:
                    by_octant = True
                    # In this case, the quas(/oc)tree is no more local into
                    # the current process, so we have to find it globally.
                    index = get_ghost_global_idx(neighs[0])
                    # \".index\" give us the index of \"self._global_ghosts\" 
                    # that contains the index of the global ghost quad(/oc)tree
                    # previously found and stored in \"index\".
                    py_ghost_oct = get_ghost_octant(neighs[0])
                    m_index = mask_octant(index + g_d)
                if (m_index != -1):
                    cell_center = get_center(py_ghost_oct,
                                             by_octant)[: dimension]
                    centers.append(cell_center)
                    indices.append(m_index)
            # ...we need to evaluate boundary values (background) or not to 
            # consider the indeces and centers found (foreground).
            else:
                to_consider = True
                border_center = neighbour_centers(c_c  ,
                                                  codim,
                                                  face_node)

                if (not is_background):
                    threshold = 0.0
                    to_consider = False
                    t_center =  apply_persp_trans(dimension    , 
                                                  border_center, 
                                                  c_t_dict     ,
                                                  logger       ,  
                                                  log_file)[: dimension]
                    check = is_point_inside_polygon(t_center    ,
                                                    t_background,
                                                    logger      ,
                                                    log_file    ,
                                                    threshold)
                    if (not check):
                        to_consider = True
                    
                if (to_consider):
                    centers.append(border_center)
                    indices.append("outside_bg")

        return (centers, indices)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Apply restriction/prolongation operators.
    def apply_rest_prol_ops(self       ,
                            row_indices,
                            col_indices,
                            col_values ,
                            centers):
        """Method which applies the right coefficients at the right neighbours
           in the prolongaion and restriction blocks.

           Arguments: 
                row_indices (list) : indices of the rows where to apply the
                                     coefficients.
                col_indices (list) : indices of the columns where to apply the
                                     coefficients.
                col_values (list) : elements to insert into \"col_indices\"."""

        grid = self._proc_g
        is_background = True
        dimension = self._dim
        # Current transformation matrix's dictionary.
        c_t_dict = self.get_trans(grid)
        if grid:
            is_background = False
            numpy_row_indices = numpy.array(row_indices)
            numpy_row_indices = numpy_row_indices[numpy_row_indices >= 0]
        insert_mode = PETSc.InsertMode.ADD_VALUES
        n_rows = 1 if is_background else numpy_row_indices.size
        to_rhs = []
        # Exact solutions.
        e_sols = []

        for i, index in enumerate(col_indices):
            # If the neighbour is outside the background boundary, the exact
            # solution is evaluated.
            if index == "outside_bg":
                to_rhs.append(i)
                e_sol = ExactSolution2D.ExactSolution2D.solution(centers[i][0]     ,
                                                                 centers[i][1]     , 
                                                                 centers[i][2] if\
                                                                 (dimension == 3)\
                                                                 else None         ,
                                                                 mapping = c_t_dict)
                e_sols.append(e_sol)

        for i in range(0, n_rows):
            row_index = row_indices if is_background else numpy_row_indices[i]
            co_indices = col_indices
            co_values = col_values
            if not is_background:
                row_index = self._ngn[row_index]
            # If \"to_rhs\" is not empty.
            if not not to_rhs:
                bil_coeffs = [col_values[j] for j in to_rhs]
                for i in range(0, len(to_rhs)):
                    self._rhs.setValues(row_index                       ,
                                        (-1 * bil_coeffs[i] * e_sols[i]),
                                        insert_mode)
                
                co_indices = [col_indices[j] for j in 
                               range(0, len(col_indices)) if j not in to_rhs]
                co_values = [col_values[j] for j in 
                              range(0, len(col_values)) if j not in to_rhs]
                
            self._b_mat.setValues(row_index  ,
                                  co_indices ,
                                  co_values  ,
                                  insert_mode)

        msg = "Applied prolongation and restriction operators."
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    def evaluate_norms(self          , 
                       exact_solution,
                       solution):
        """Function which evals the infinite and L2 norms of the error.

           Arguments:
                exact_solution (numpy.array) : exact solution.
                solution (numpy.array) : computed solution.

           Returns:
                (norm_inf, norm_L2) (tuple of int): evaluated norms."""

        h = self._h
        octant_area = (h * h)
        numpy_difference = numpy.subtract(exact_solution,
                                          solution)
        norm_inf = numpy.linalg.norm(numpy_difference,
                                     # Type of norm we want to evaluate.
                                     numpy.inf)
        norm_L2 = numpy.linalg.norm(numpy_difference,
                                    2) * h

        msg = "Evaluated norms"
        extra_msg = "with (norm_inf, norm_L2) = " + str((norm_inf, norm_L2))
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        return (norm_inf, norm_L2)
    # --------------------------------------------------------------------------
        
    
    @property
    def comm(self):
        return self._comm

    @property
    def octree(self):
        return self._octree

    @property
    def N(self):
        return self._N_oct

    @property
    def n(self):
        return self._n_oct

    @property
    def mat(self):
        return self._b_mat

    @property
    def rhs(self):
        return self._rhs

    @property
    def sol(self):
        return self._sol

    @property
    def h(self):
        return self._h

    @property
    def not_pen_centers(self):
        return self._centers_not_penalized

