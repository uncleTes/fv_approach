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
        if (grid):
            is_background = False
        # If we are on the background grid, we need to re-evaluate the ranges
        # of octants owned by each process, not simply adding the masked ones
        # (as we will do for the foreground grids) to the values given by PETSc
        # function \"getOwnershipRange\" on the matrix. The idea is to take as
        # start of the range the sum of all the octants owned by the previous
        # processes of the current process, while for the end of the range
        # take the same range of processes plus the current one, obviously
        # subtracting the value \"1\", because the octants start from \"0\".
        if (is_background):
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
        narray = numpy.array

        for octant in xrange(0, n_oct):
            h2 = octree.get_area(inter        ,
                                 is_ptr = True,
                                 is_inter = False)
            h = numpy.sqrt(h2)
            # Global index of the current local octant \"octant\".
            g_octant = o_ranges[0] + octant
            m_g_octant = mask_octant(g_octant)
            # Check if the octant is not penalized.
            if (m_g_octant != -1):
                py_oct = get_octant(octant)
                center  = get_center(octant)[: dimension]

                # Lambda function.
                g_b = lambda x : get_bound(py_oct, x)

                for face in xrange(0, nfaces):
                    # If we have an edge on the boundary.
                    if (g_b(face)):
                        b_indices.append(m_g_octant)
                        b_f_o_n.append(face)
                        b_centers.append(center)
                        b_codim.append(1)

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
                numpy_b_center = narray(b_center)
                t_b_center =  apply_persp_trans(dimension     ,
                                                numpy_b_center,
                                                c_t_dict)[: dimension]
                prev_t_b_center = t_b_center

            t_b_centers.append(t_b_center)

        # Grids not of the background: equal to number >= 1.
        if (grid):
            for i, center in enumerate(c_neighs):
                check = False
                # Check if the \"ghost\" points outside the foreground grids are
                # inside the background one.
                numpy_center = narray(center)
                t_center, n_t_center =  apply_persp_trans(dimension   ,
                                                          numpy_center,
                                                          c_t_dict    ,
                                                          True)[: dimension]
                check = is_point_inside_polygon(t_center    ,
                                                t_background)
                if (check):
                    n_axis = 0 if (b_f_o_n[i] < 2) else 1
                    # Can't use list as dictionary's keys.
                    # http://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python
                    # https://wiki.python.org/moin/DictionaryKeys
                    key = (grid        , # Grid to which the index belongs to
                           b_indices[i], # Masked global index of the octant
                           n_axis)
                    # We store the center of the cell on the boundary.
                    # TODO: why we store also \"center\" and not just 
                    #       \"b_centers[i]\"? Think about it. And why add also
                    #       \"h\": is not useless in a fv approach?
                    #       And it is not useless also \"b_centers[i]\"?
                    t_value = (h,) + tuple(center[: dimension])
                    t_value = t_value + tuple(b_centers[i][: dimension])
                    # Length of the stencil.
                    l_stencil = 36 if (dimension == 2) else 45
                    n_mask = l_stencil - len(t_value)
                    stencil = (t_value + ((-1,) * (n_mask)))
                    self._edl.update({key : stencil})
                    # The new corresponding value inside \"b_values\" would be
                    # \"0.0\", because the boundary value is given by the 
                    # coefficients of the bilinear operator in the \"extension\"
                    # matrix.
                    b_values[i] = 0.0

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
    def check_neighbours(self         ,
                         codim        ,
                         f_o_n        ,
                         octant       ,
                         o_count      ,
                         d_count      ,
                         # Stencil's index.
                         s_i          ,
                         key          ,
                         is_penalized ,
                         is_background):
        logger = self.logger
        log_file = logger.handlers[0].baseFilename
        dimension = self._dim
        octree = self._octree

        # Code hoisting.
        get_nodes = octree.get_nodes
        get_center = octree.get_center
        get_global_idx = octree.get_global_idx
        get_ghost_global_idx = octree.get_ghost_global_idx
        get_ghost_octant = octree.get_ghost_octant
        find_neighbours = octree.find_neighbours
        check_oct_corners = self.check_oct_corners
        mask_octant = self.mask_octant
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons

        # Check to know if a neighbour of an octant is penalized.
        is_n_penalized = False
        neighs, ghosts = ([] for i in range(0, 2))

        neighs, \
        ghosts = find_neighbours(octant,
                                 f_o_n ,
                                 codim ,
                                 neighs,
                                 ghosts)
        # Number of neighbours.
        n_neighbours = len(neighs)

        # Being in a case of a possible jump of 1 level between elements, we
        # have to consider two possible neighbours for each face of the octant.
        for i in xrange(0, n_neighbours):
            if (not ghosts[i]):
                    index = get_global_idx(neighs[i])
                    n_center = get_center(neighs[i])[: dimension]
            else:
                index = get_ghost_global_idx(neighs[i])
                py_ghost_oct = get_ghost_octant(neighs[i])
                n_center = get_center(py_ghost_oct,
                                      True)[: dimension]
            if (is_background):
                t_foregrounds = self._t_foregrounds
                # Current transformation matrix's dictionary.
                c_t_dict = self.get_trans(0)
                idx_or_oct = neighs[i] if (not ghosts[i]) else \
                             py_ghost_oct
                is_ptr = False if (not ghosts[0]) else \
                         True

                oct_corners, \
                numpy_corners = get_nodes(idx_or_oct,
                                          dimension ,
                                          is_ptr    ,
                                          also_numpy_nodes = True)

                is_n_penalized, \
                n_polygon = check_oct_corners(numpy_corners,
                                              c_t_dict     ,
                                              t_foregrounds)
            if (not is_penalized):
                if (is_n_penalized):
                    # Being the neighbour penalized, it means that it will
                    # be substituted by 9 octant being part of the fore-
                    # ground grids, so being on the non diagonal part of the
                    # grid.
                    o_count += 9
                else:
                    if (ghosts[i]):
                        o_count += 1
                    else:
                        d_count += 1
            else:
                if (not is_n_penalized):
                    stencil = self._edl.get(key)
                    stencil[s_i] = index
                    stencil[s_i + 1 : s_i + (dimension + 1)] = n_center
                    s_i += 4 if (dimension == 2) else 5

            extra_msg = ""

            msg = "Checked neighbour for "               + \
                  ("edge " if (codim == 1) else "node ") + \
                  str(index)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
        
        return (d_count,
                o_count,
                s_i    ,
                n_neighbours)
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
        dimension = self._dim
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
        get_nodes = octree.get_nodes
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons
        get_bound = octree.get_bound
        check_neighbours = self.check_neighbours
        check_oct_corners = self.check_oct_corners

        for octant in octants:
            d_count, o_count = 0, 0
            g_octant = g_octants[octant]
            py_oct = py_octs[octant]
            h2 = octree.get_area(py_oct       ,
                                 is_ptr = True,
                                 is_inter = False)
            h = numpy.sqrt(h2)
            # Lambda function.
            g_b = lambda x : get_bound(py_oct,
                                       x)
            center  = centers[octant]
            # Check to know if an octant is penalized.
            is_penalized = False
            # Background grid.
            if (is_background):
                oct_corners, \
                numpy_corners = get_nodes(octant   ,
                                          dimension,
                                          also_numpy_nodes = True)

                is_penalized, \
                n_polygon = check_oct_corners(numpy_corners,
                                              c_t_dict     ,
                                              t_foregrounds)
            if (is_penalized):
                self._nln[octant] = -1
                # Moved \"h\" from the \"key\" to the \"stencil\", preferring
                # not to use float into dict keys.
                key = (n_polygon, # Foreground grid to which the node belongs to
                       g_octant,  # Global index (not yet masked)
                       0)         # Useless field, use to pair with the \"key\"
                                  # for foreground grids.
                # If the octant is covered by the foreground grids, we need to
                # store info of the stencil it belongs to to push on the rela-
                # tive rows of the matrix, the right indices of the octants of
                # the foreground grid owning the penalized one:
                # first \"stencil\"'s element: \"h\";
                # second \"stencil\"'s element: index global of the penalized
                # octant;
                # third, fourth and fifth \"stencil\"'s elements: center of the
                # penalized octant;
                # others \"stencil\"'s elements: global indices and centers of
                # the neighbours not penalized (being in a case of a possible
                # jump of 1 level between elements, we have to store two possi-
                # ble neighbours for each face of the current octant), and coef-
                # ficient to multiply \"least squares\" approximation.
                stencil = [-1] * 36 if (dimension == 2) else [-1] * 45
                stencil[0] = h # TODO: is this useful or not? I think not.
                stencil[1] = g_octant
                stencil[2 : (dimension + 1)] = center # TODO: is this useful or
                                                      # not? I think not, in a
                                                      # fv approach.
                # http://www.laurentluce.com/posts/python-dictionary-implementation/
                # http://effbot.org/zone/python-hash.htm
                self._edl.update({key : stencil})
            else:
                self._nln[octant] = new_oct_count
                new_oct_count += 1
                d_count += 1
            # \"stencil\"'s index.
            s_i = 4 if (dimension == 2) else 5
            # Number of neighbours (Being the possibility of a jump between oc-
            # tants, we can have a minimum of 4 and a maximum of 8 neighbours.
            n_neighbours = 0
            # Faces' loop.
            for face in xrange(0, nfaces):
                # Not boundary face.
                if (not g_b(face)):
                    d_count,
                    o_count,
                    s_i    ,
                    n_neighs = check_neighbours(1                            ,
                                                face                         ,
                                                octant                       ,
                                                o_count                      ,
                                                d_count                      ,
                                                s_i                          ,
                                                key if is_penalized else None,
                                                is_penalized                 ,
                                                is_background)
                    n_neighbours += n_neighs
                else:
                    n_neighbours += 1

                    if (not is_background):
                        # Adding elements for the octants of the background to
                        # use to interpolate stencil values for boundary condi-
                        # tions of the octants of the foreground grid. Note that
                        # we are considering the fact that the foreground grids
                        # elements are contained into the background, and we are
                        # not checking it or if they are outside it (it should
                        # not have a lot of sense however). That's why we are
                        # adding always +9 to the \"o_count\" variable, despite
                        # the fact that they could be on the border of the back-
                        # ground domain, exactly, and in this case we should not
                        # add anything to \"o_count\" and \"d_count\", being an
                        # exact boundary condition.
                        # TODO: write a better way to modify \"d_count\" and
                        # \"o_count\".
                        o_count += 9

            # For the moment, we have to store space in the \"PETSc\" matrix for
            # the octants that will interpolate with the least square method (9
            # octants in 2D at maximum) for the vertices of each intersection.
            # And these vertices are equal to the number of neighbours of the
            # current octant (With a gap of one level, we can have as maximum
            # two neighbours for each face), but we do not know a priori if the
            # owners of the nodes will be ghosts (\"o_count\"), or not
            # (\"d_count\").
            # TODO: find a better algorithm to store just the right number of e-
            # lements for \"d_count\" and for \"o_count\".
            d_count += (9 * n_neighbours)
            o_count += (9 * n_neighbours)
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

    # Get global indices of the owners of the outer/inner normals of the
    # intersection.
    def get_owners_normals_inter(self ,
                                 inter,
                                 is_ghost_inter):
        octree = self._octree
        # Is owner outer normal ghost.
        is_o_o_n_g = False
        # Index owner outer normal.
        i_o_o_n = octree.get_out(inter)
        # Index owner inner normal.
        i_o_i_n = octree.get_in(inter)
        # Global index owner inner normal.
        g_i_o_i_n = None
        # Global index owner outer normal.
        g_i_o_o_n = None
        # Index of the owner ghost (if present): 1 for the one with outer nor-
        # mal, 0 for the inner normal.
        o_ghost = None

        if (is_ghost_inter):
            is_o_o_n_g = octree.get_out_is_ghost(inter)
            if (is_o_o_n_g):
                o_ghost = 1
                g_i_o_o_n = octree.get_ghost_global_idx(i_o_o_n)
                g_i_o_i_n = octree.get_global_idx(i_o_i_n)
            else:
                o_ghost = 0
                g_i_o_i_n = octree.get_ghost_global_idx(i_o_i_n)
                g_i_o_o_n = octree.get_global_idx(i_o_o_n)
        else:
            g_i_o_o_n = octree.get_global_idx(i_o_o_n)
            g_i_o_i_n = octree.get_global_idx(i_o_i_n)

        return [[g_i_o_i_n, g_i_o_o_n],
                [i_o_i_n, i_o_o_n]    ,
                o_ghost]

    def get_interface_coefficients(self          ,
                                   inter         , # pointer to the intersection
                                   dimension     , # 2D/3D
                                   nodes_inter   , # Coordinates of the nodes
                                                   # of the intersection
                                   owners_center , # Centers of the owners of
                                                   # the intersection
                                   l_s_coeffs    , # Least square coefficients.
                                   is_bound_inter, # is a boundary intersection
                                   n_normal_inter, # numpy normal vector to the
                                                   # intersection
                                   n_axis):        # directional axis of the
                                                   # Normal (0 for x, 1 for y)
        # evaluating length of the intersection, depending on its direc-
        # tion.
        h = octree.get_area(inter        ,
                            is_ptr = True,
                            is_inter = True)

        d_nodes_x    , \
        d_nodes_y    , \
        c_inter      , \
        d_o_centers_x, \
        d_o_centers_y,  =  self.get_interface_distances(dimension     ,
                                                        nodes_inter   ,
                                                        is_bound_inter,
                                                        n_axis)
        # Normal is parallel to y-axis.
        if (n_axis):
            temp = d_o_centers_x
            d_o_centers_x = d_o_centers_y
            d_o_centers_y = temp
            d_nodes_y = d_nodes_x

        coeff_in = 1.0 / d_o_centers_x
        coeff_out = -1.0 * coeff_in
        coeff_node_1 = (-1.0 * d_o_centers_y) / \
                       (d_o_centers_x * d_nodes_y)
        coeff_node_0 = -1.0 * coeff_node_1
        # \"Numpy\" coefficients.
        n_coeffs = numpy.array([coeff_in    ,
                                coeff_out   ,
                                coeff_node_1,
                                coeff_node_0])
        # Multiplying \"numpy\" coefficients for the normal to the in-
        # tersection and for the length of the intersection.
        n_coeffs = n_coeffs * n_normal_inter[n_axis] * h

        coeffs_node_1 = l_s_coeffs[1] * n_coeffs[2]
        coeffs_node_0 = l_s_coeffs[0] * n_coeffs[3]

        return (n_coeffs     ,
                coeffs_node_1,
                coeffs_node_0)

    def get_interface_distances(self          ,
                                dimension     ,
                                nodes_inter   ,
                                is_bound_inter,
                                n_axis):
        # Distance between xs of nodes of the intersection (greater
        # coordinates are in the second (remember, we are now in 2D)
        # value of \"normal_nodes\").
        d_nodes_x = numpy.absolute(nodes_inter[1][0] - \
                                   nodes_inter[0][0])
        # Distance between ys of nodes of the intersection (greater
        # coordinates are in the second (remember, we are now in 2D)
        # value of \"normal_nodes\").
        d_nodes_y = numpy.absolute(nodes_inter[1][1] - \
                                   nodes_inter[0][1])
        # Center of the intersection.
        c_inter = ((nodes_inter[1][0] + nodes_inter[0][0]) / 2.0,
                   (nodes_inter[1][1] + nodes_inter[0][1]) / 2.0)

        d_o_centers_x = 0.0
        d_o_centers_y = 0.0
        if (is_bound_inter):
            # Normal parallel to y-axis.
            if (n_axis):
                # Distance between y of center of the octant owner of
                # the intersection and the extern boundary.
                d_o_centers_y = h
            # Normal parallel to x-axis.
            else:
                # Distance between x of center of the octant owner of
                # the intersection and the extern boundary.
                d_o_centers_x = h
        else:
            # Distance between xs of centers of the octants partaging
            # the intersection.
            d_o_centers_x = numpy.absolute(owners_centers[1][0] - \
                                           owners_centers[0][0])
            # Distance between ys of centers of the octants partaging
            # the intersection.
            d_o_centers_y = numpy.absolute(owners_centers[1][1] - \
                                           owners_centers[0][1])

            return (d_nodes_x    ,
                    d_nodes_y    ,
                    c_inter      ,
                    d_o_centers_x,
                    d_o_centers_y)


    def get_l_owners_nodes_inter(self          ,
                                 inter         ,
                                 l_owners_inter,
                                 o_ghost       , # Owner ghost; if it is \"No-
                                                 # ne\", there is no owner ghost
                                                 # but, if it is \"0\" or \"1\",
                                                 # it means that is respectively
                                                 # the owner with the inner nor-
                                                 # mal or the one with the outer
                                                 # one.
                                 also_nodes = False): # Return also the nodes of
                                                      # the intersection, not
                                                      # just their owners
        octree = self._octree
        tot_oct = self._tot_oct
        dimension = self._dim
        n_nodes = 2 if (dimension == 2) else 4
        nodes = octree.get_nodes(inter        ,
                                 dimension    ,
                                 is_ptr = True,
                                 is_inter = True)[: n_nodes]
        # Local indices of the octants owners of the nodes of the
        # intersection.
        l_owners = [0] * n_nodes
        for i in xrange(0, n_nodes):
            # Returning \"max uint32_t\" if point outside of the domain.
            #
            # Temp owner.
            t_owner = octree.get_point_owner_idx(nodes[i])
            # If the index of the owner if bigger than the total number of
            # octants present in the problem, we have reached or a ghost
            # octant or we are outside the domain.
            if (t_owner > tot_oct):
                if (o_ghost is not None):
                    # If the intersection is ghost, then we have only one lo-
                    # cal octant owner.
                    l_owner = l_owners_inter[1 - o_ghost]
                else:
                    # In this case, the owner will be one of the owners of the
                    # intersection (it will be different for each node in 2D,
                    # and different in \"modulo 2\" in 3D, excpet for boundary
                    # intersection where the local onwer will be always the sa-
                    # me).
                    l_owner = l_owners_inter[i % 2]
            else:
                #l_owner = octree.get_global_idx(l_owner)
                l_owner = t_owner
            l_owners[i] = l_owner

        if (also_nodes):
            return (l_owners, nodes)

        return l_owners

    def get_owners_inter(self          ,
                         inter         ,
                         starting_index,
                         is_background ,
                         is_ghost_inter):
        octree = self._octree
        # Indices of the owners of the intersection.
        owners_inter = octree.get_owners(inter)
        # Global indices of the owners.
        g_owners_inter = []
        g_owner = starting_index + owners_inter[0]
        g_owners_inter.append(g_owner)

        if (is_ghost_inter):
            g_owner = octree.get_ghost_global_idx(owners_inter[1])
            if (not is_background):
                g_owner += starting_index
        else:
            g_owner = starting_index + owners_inter[1]
        g_owners_inter.append(g_owner)

        return g_owners_inter

    # --------------------------------------------------------------------------
    # Initialize diagonal matrices of the block matrix.
    def init_mat(self,
                 (d_nnz, o_nnz)):
        """Method which initialize the diagonal parts of the monolithic matrix 
           of the system.
           
           Arguments:
               (d_nnz, o_nnz) (tuple) : two lists containting the diagonal and
                                            non diagonal block's number of non
                                            zero elements, passed by function
                                            \"create_mask\"."""

	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        f_bound = self._f_bound
        grid = self._proc_g

        # Ghosts' deplacement.
        g_d = 0
        for i in xrange(0, grid):
            g_d = g_d + self._oct_f_g[i]

        comm_w = self._comm_w
        rank_w = self._rank_w
        octree = self._octree
        tot_oct = self._tot_oct
        is_background = False if (grid) else True
        t_foregrounds = self._t_foregrounds

        n_oct = self._n_oct
        nfaces = octree.get_n_faces()
        ninters = octree.get_num_intersections()
        face_node = octree.get_face_node()
        dimension = self._dim
        sizes = self.find_sizes()

        self._b_mat = PETSc.Mat().createAIJ(size = (sizes, sizes),
                                            nnz = (d_nnz, o_nnz) ,
                                            comm = comm_w)
        # If an element is being allocated in a place not preallocate, then 
        # the program will stop.
        self._b_mat.setOption(self._b_mat.Option.NEW_NONZERO_ALLOCATION_ERR, 
                              True)
        o_ranges = self.get_ranges()

        # Current transformation matrix's dictionary.
        c_t_dict = self.get_trans(grid)
        # Current transformation adjoint matrix's dictionary.
        c_t_a_dict = self.get_trans_adj(grid) 
        
        # Code hoisting.
        mask_octant = self.mask_octant
        get_octant = octree.get_octant
        get_ghost_octant = octree.get_ghost_octant
        get_center = octree.get_center
        get_nodes = octree.get_nodes
        get_intersection = octree.get_intersection
        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons
        metric_coefficients = utilities.metric_coefficients
        square = numpy.square
        check_neighbours = self.check_neighbours
        get_bound = octree.get_bound
        narray = numpy.array
        check_oct_corners = self.check_oct_corners
        get_owners_inter = self.get_owners_inter
        get_owners_normals_inter = self.get_owners_normals_inter
        get_owners_nodes_inter = self.get_owners_nodes_inter
        get_is_ghost = octree.get_is_ghost
        least_squares = utilities.least_squares
        # Lambda functions.
        g_n = lambda x : get_nodes(x               ,
                                   dimension       ,
                                   is_ptr = True   ,
                                   is_inter = False,
                                   also_numpy_nodes = True)
        f_r_n = lambda x : find_right_neighbours(x          ,
                                                 o_ranges[0],
                                                 is_background)
        l_s = lambda x : least_squares(x[0],
                                       x[1])

        for i in xrange(0, ninters):
            # Rows and columns indices for the \"PETSC\" matrix.
            r_indices, \
            c_indices = ([] for i in range(0, 2))
            # Centers of the owners of the intersection.
            owners_centers = []
            # Centers and indices of the neighbours of the octants owners of
            # the nodes of the intersection.
            neigh_centers, \
            neigh_indices = ([] for i in range(0, 2))
            inter = get_intersection(i)
            # Is a ghost intersection.
            is_ghost_inter = get_is_ghost(inter,
                                          True) # Using intersection
                                                # instead of octant
            # Is a boundary intersection.
            is_bound_inter = get_bound(inter,
                                       0    , # Being with an intersection, it
                                       True)  # does not matter what number we
                                              # are giving to the second arg
            # Global indices of owners inner/outer normal of the intersection
            # (is a list, and the first element is the one with the inner nor-
            # mal), followed by local indices of owners, and a list of boolean
            # to know if and what owners are ghost.
            #
            # Here, global means global in the current octree. To have global
            # for the totality of the octrees, we have to add \"g_d\".
            #
            # First owner will be the one with the inner normal.
            #
            # Owner ghost equal to 1 means that is ghost the one with the outer
            # normal, if is 0 is ghost the one with the inner.
            g_o_norms_inter,
            l_o_norms_inter,
            o_ghost = get_owners_normals_inter(inter,
                                               is_ghost_inter)
            # List containing 0 or 1 to indicate inner normal or outer normal.
            labels = []
            # Masked global indices of owners inner/outer normal of the
            # intersection.
            m_g_o_norms_inter = map(mask_octant,
                                    [(g_o_norm_inter + g_d) for g_o_norm_inter \
                                     in g_o_norms_inter])
            # Normal to the intersection, and its numpy version.
            normal_inter, \
            n_normal_inter = pablo.get_normal(inter,
                                              True) # We want also
                                                    # a \"numpy\"
                                                    # version
            # Normal's axis, indicating the non-zero component of the
            # normal.
            n_axis = numpy.nonzero(n_normal_inter)[0][0]
            # Number of intersection's owners (both in 2D and 3D, it will be
            # always equal to 2).
            n_i_owners = 2
            # Looping on the owners of the intersection.
            for j in xrange(0, n_i_owners):
                # Here, means that the owner is or ghost or outside the local
                # domain of the octree, so we do not want to do operations on
                # that one.
                if (j == o_ghost):
                    py_oct = get_ghost_octant(l_o_norms_inter[j])
                else:
                    py_oct = get_octant(l_o_norms_inter[j])
                center, \
                numpy_center = get_center(py_oct           ,
                                          ptr_octant = True,
                                          also_numpy_center = True)[: dimension]
                owners_centers.append(numpy_center)
                m_g_octant = m_g_o_norms_inter[j]
                is_penalized = False
                # If an intersection owner is penalized (it could be just for
                # background grid)...
                if (m_g_octant == -1):
                    is_penalized = True

                    oct_corners,
                    numpy_corners = g_n(py_oct)
                    # \"is_penalized_useless\" will not be used because we al-
                    # ready know that the octant is penalized (we entered the
                    # \"if\" clause). But we need \"n_polygon\" to be used in
                    # the \"key\" for the background grid.
                    is_penalized_useless,
                    n_polygon = check_oct_corners(numpy_corners,
                                                  c_t_dict     ,
                                                  t_foregrounds)
                # ...else...
                if (not is_penalized):
                    r_indices.append(m_g_octant)
                    if (is_bound_inter):
                        # Normal always directed outside the domain.
                        labels.append(1)
                    else:
                        labels.append(j)
            if (is_bound_inter and (len(r_indices) == 2)):
                # Being a boundary intersection, owner is the same.
                del r_indices[-1]
            # If the owners of the intersection are not both covered.
            if (r_indices):
                # Local indices of the octants owners of the nodes of the
                # intersection (needed for \"find_right_neighbours\"
                # function). Here, local means global but in the current octant.
                l_o_nodes_inter, \
                # The coordinates of the nodes.
                nodes_inter = get_l_owners_nodes_inter(inter          ,
                                                       l_o_norms_inter,
                                                       o_ghost        ,
                                                       also_nodes = True)
                # Neighbour centers neighbours indices: it is a list of
                # tuple, and in each tuple are contained the lists of
                # centers and indices of each local owner of the nodes.
                n_cs_n_is = map(f_r_n,
                                l_o_nodes_inter)
                # Least square coefficients.
                l_s_coeffs = map(l_s,
                                 zip([pair[0] for pair in n_cs_n_is],
                                     [narray(node) for node in nodes_inter]))

                n_coeffs     , \
                coeffs_node_1, \
                coeffs_node_0,  =  self.get_interface_coefficients(inter         ,
                                                                   dimension     ,
                                                                   nodes_inter   ,
                                                                   owners_center ,
                                                                   l_s_coeffs    ,
                                                                   is_bound_inter,
                                                                   n_normal_inter,
                                                                   n_axis)

                if (is_ghost_inter and (len(r_indices) == 2)):
                    # Using \"extend.([number])\" to avoid \"TypeError: 'int'
                    # object is not iterable\" error.
                    c_indices.extend([r_indices[1 - o_ghost]])
                    # The owner of the inner normal will add values also for the
                    # nodes of the intersection.
                    if (o_ghost == 1):
                        c_indices.extend(n_cs_n_is[1][1])
                        c_indices.extend(n_cs_n_is[0][1])
                else:
                    c_indices.extend(r_indices)
                    c_indices.extend(n_cs_n_is[1][1])
                    c_indices.extend(n_cs_n_is[0][1])
                # Both the owners of the intersection are not penalized.
                if (len(r_indices) == 2):
                    # Values to insert in \"r_indices\"; each sub list contains
                    # values for each owner of the intersection.
                    values = [[], []]
                    # \"Numpy\" temporary array.
                    n_t_array = numpy.array([n_coeffs[0],
                                             n_coeffs[1]])
                    # If is a ghost intersection, we store just one coefficient
                    # for each time we will pass on the current intersection
                    # (being ghost, there will be two processes owning it).
                    if (is_ghost_inter):
                        n_t_array = numpy.array([n_coeffs[1 - o_ghost])
                    # So, if there is no ghost intersection or if the ghost is
                    # the owner of the outer normal.
                    if (o_ghost != 0): 
                        n_t_array = numpy.append(n_t_array,
                                                 coeffs_node_1)
                        n_t_array = numpy.append(n_t_array,
                                                 coeffs_node_0)
                    # \"values[0]\" is for the owner with the inner normal,
                    # while \"values[1]\" is for the owner with the outer one:
                    # Add to the octant with the outer normal, subtract to the
                    # one with the inner normal.
                    values[1] = n_t_array.tolist()
                    values[0] = (n_t_array * -1).tolist()
                # Just one owner is not penalized, or we are on the boundary.
                else:
                    # Values to insert in \"r_indices\".
                    values = []
                    n_t_array = numpy.array([n_coeffs[labels[0]]])
                    n_t_array = numpy.append(n_t_array,
                                             coeffs_node_1)
                    n_t_array = numpy.append(n_t_array,
                                             coeffs_node_0)
                    # Here we can be only on the background, where some octants
                    # are penalized.
                    if (not is_bound_inter):
                        mult = -1.0
                        # Owner with the outer normal is not penalized, so we
                        # have to add the coefficients in the corresponding row,
                        # instead of subtract them.
                        if (labels[0]):
                            mult = 1.0
                        # Penalized global index, not masked.
                        p_g_index = g_o_norms_inter[1 - labels[0]]
                        value_to_store = n_coeffs[1 - labels[0]] * mult

                        key = (n_polygon,
                               p_g_index,
                               0)

                        stencil = self._edl.get(key)
                        step = 4 if (dimension == 2) else 5
                        for (k in stencil[step::step]):
                            if (stencil[k] == p_g_index):
                                stencil[k + dimension + 1] = value_to_store
                    # We are on a boundary intersection; here normal is always
                    # directed outside, so the owner is the one with the outer
                    # normal.
                    else:
                        m_octant = m_g_o_norms_inter[labels[0]]
                        mult = 1.0
                        value_to_store = n_coeffs[1 - labels[0]] * mult
                        if (is_background):
                            # The multiplication for \"(-1 * mult)\" is due to
                            # the fact that we have to move the coefficients to
                            # the \"rhs\", so we have to change its sign.
                            value_to_store = value_to_store * -1
                            # Previous \"rhs\" value.
                            p_rhs_v = self._rhs[m_octant]
                            self._rhs.setValues(m_octant                ,
                                                p_rhs_v * value_to_store,
                                                PETSc.InsertMode.INSERT_VALUES)
                        else:
                            key = (grid      ,
                                   m_g_octant,
                                   n_axis)
                            stencil = self._edl.get(key)
                            stencil[(dimension * 2) + 1] = value_to_store

                    values = (n_t_array * mult).tolist()

                self._b_mat.setValues(r_indices, # Row
                                      c_indices, # Columns
                                      values)    # Values to be inserted
    
        # We have inserted argument \"assebly\" equal to 
        # \"PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY\" because the final assembly
        # will be done after inserting the prolongation and restriction blocks.
        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY)
        self.assembly_petsc_struct("rhs")
        
        msg = "Initialized diagonal parts of the monolithic  matrix"
        extra_msg = "with sizes \"" + str(self._b_mat.getSizes()) + \
                    "\" and type \"" + str(self._b_mat.getType()) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------

    def check_oct_corners(self         ,
                          numpy_corners,
                          t_dict       ,
                          polygons):
        penalized = True
        dimension = self._dim
        n_oct_corners = 4 if (dimension == 2) else 8

        apply_persp_trans = utilities.apply_persp_trans
        is_point_inside_polygons = utilities.is_point_inside_polygons

        for i in xrange(n_oct_corners):
            is_corner_penalized = False
            numpy_corner = numpy_corners[i]
            corner, numpy_corner = apply_persp_trans(dimension   ,
                                                     numpy_corner,
                                                     t_dict      ,
                                                     # Return also numpy data
                                                     r_a_n_d = True)
            (is_corner_penalized,
             n_polygon) = is_point_inside_polygons(numpy_corner,
                                                   polygons)
            if (not is_corner_penalized):
                penalized = False
                break

        return (penalized, n_polygon)

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
        if (not grid):
            # Not masked local octant background grid.
            not_masked_l_oct_bg_g = numpy.size(self._nln[self._nln != -1])
        sizes = (n_oct if (grid) else not_masked_l_oct_bg_g,
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

    def add_rhs(self,
                numpy_array):
        self._rhs = self.add_array("right hand side",
                                   True             ,
                                   self._rhs        ,
                                   numpy_array)
        msg = "Added array to \"rhs\""
        self.log_msg(msg,
                     "info")

    def add_array(self             ,
                  a_name = ""      ,
                  petsc_size = True,
                  vec              ,
                  array_to_add):
        if not petsc_size:
            n_oct = self._n_oct
            tot_oct = self._tot_oct
            sizes = (n_oct, tot_oct)
        else: 
            sizes = self.find_sizes()
        try:
            assert isinstance(array_to_add, numpy.ndarray)
            # Temporary PETSc vector.
            t_petsc = PETSc.Vec().createWithArray(array_to_add,
                                                  size = sizes,
                                                  comm = self._comm_w)
            vec.axpy(1.0, t_petsc)
        except AssertionError:
            msg = "\"MPI Abort\" called during array's initialization"
            extra_msg = "Parameter \"array\" not an instance of " + \
                        "\"numpy.ndarray\"."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
            self._comm_w.Abort(1)
        msg = "Added array to \"" + str(a_name) + "\""
        self.log_msg(msg,
                     "info")
        return vec
   
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
                 numpy_array = None):
	"""Method which intializes the right hand side.
            
           Arguments:
                numpy_array (numpy.array) : array to use to initialize with it
                                            the \"rhs\"."""

        self._rhs = self.init_array(a_name = "right hand side",
                                    petsc_size = True         ,
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
        
        msg = "Initialized \"solution\""
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

        dimension = self._dim
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
        # TODO: check to see if is better to use int64 or uint64.
        if not is_background:
            self._d_type_s = numpy.dtype('(1, 3)i8, (1, 36)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 3)i8, (1, 45)f8')
            blocks_length_s = [3, 36] if (dimension == 2) else [3, 45]
            blocks_displacement_s = [0, 24]
            mpi_datatypes = [MPI.INT64_T,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 3)i8, (1, 36)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 3)i8, (1, 45)f8')
            blocks_length_r = [3, 36] if (dimension == 2) else [3, 45]
            blocks_displacement_r = [0, 24]
        else:
            self._d_type_s = numpy.dtype('(1, 3)i8, (1, 36)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 3)i8, (1, 45)f8')
            blocks_length_s = [3, 36] if (dimension == 2) else [3, 45]
            blocks_displacement_s = [0, 24]
            mpi_datatypes = [MPI.INT64_T,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 3)i8, (1, 36)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 3)i8, (1,45)f8')
            blocks_length_r = [3, 36] if (dimension == 2) else [3, 45]
            blocks_displacement_r = [0, 24]
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
        # TODO: delete this.
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
        stencils = numpy.array([list_edg[i][1] for i in 
                                range(0, l_l_edg)]).reshape(l_l_edg, l_s)
        centers = [(stencils[i][2 : 2 + dimension]) for i in range(0, l_l_edg)]
        n_centers = len(centers)
        t_centers = [None] * n_centers

        # Code hoisting.
        apply_persp_trans = utilities.apply_persp_trans
        apply_persp_trans_inv = utilities.apply_persp_trans_inv
        find_right_neighbours = self.find_right_neighbours
        least_squares = utilities.least_squares
        metric_coefficients = utilities.metric_coefficients
        apply_rest_prol_ops = self.apply_rest_prol_ops
        narray = numpy.array

        # Lambda function.
        f_r_n = lambda x : find_right_neighbours(x, o_ranges[0])

        for i in xrange(0, n_centers):
            numpy_center = narray(centers[i])
            centers[i] = apply_persp_trans(dimension   ,
                                           numpy_center,
                                           b_t_dict)[: dimension]
            t_centers[i] = centers[i]
            numpy_center = narray(centers[i])
            centers[i] = apply_persp_trans_inv(dimension   ,
                                               numpy_center,
                                               c_t_adj_dict)[: dimension]
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
        numpy_centers = [numpy.array(center) for center in centers]
        # \"idxs[0]\" because is a numpy array, so to select the array we have
        # to use the index notation.
        for idx in idxs[0]:
            neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
            (neigh_centers, 
             neigh_indices)  = f_r_n(local_idxs[idx])
            coeffs = least_squares(neigh_centers,
                                   numpy_centers[idx])

            # Checkout how the \"stencil\" is created in the function
            # \"create_mask\".
            step = 4 if (dimension == 2) else 5
            for i in xrange(step, len(stencils[idx]), step):
                row_index = int(stencils[idx][i])
                n_center = [stencils[idx][i + 1], stencils[idx][i + 2]]
                value_to_multiply = stencils[idx][i + dimension + 1]
                numpy_n_center = narray(n_center)
                t_n_center = apply_persp_trans(dimension     ,
                                               numpy_n_center,
                                               b_t_dict)[: dimension]

                new_coeffs = [coeff * value_to_multiply for coeff in \
                              coeffs]
                apply_rest_prol_ops(row_index     ,
                                    neigh_indices ,
                                    new_coeffs,
                                    neigh_centers)

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
        # Outside centers.
        o_centers = numpy.array([list_edg[i][1][0][: dimension] for i in 
                                 range(0, l_l_edg)]).reshape(l_l_edg, dimension)
        # Inside centers.
        i_centers = numpy.array([list_edg[i][1][0][dimension : (dimension * 2)]\
                                 for i in range(0, l_l_edg)]).reshape(l_l_edg, 
                                                                      dimension)
        values_to_multiply = numpy.array([list_edg[i][1][(dimension * 2) + 1] \
                                          for i in xrange(0, l_l_edg)]).reshape(l_l_edg, dimension)
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
        narray = numpy.array

        for i in xrange(0, n_o_centers):
            f_t_dict = self.get_trans(int(keys[i][0]))
            numpy_o_center = narray(o_centers[i])
            t_o_centers[i] = apply_persp_trans(dimension     ,
                                               numpy_o_center,
                                               f_t_dict)
            numpy_t_o_center = narray(t_o_centers[i])
            t_o_centers[i] = apply_persp_trans_inv(dimension       ,
                                                   numpy_t_o_center,
                                                   c_t_adj_dict)
        # Here we need to pass \"center[0:2]\" to the function \"get_point_ow-
        # ner_dx\", while in the previous version of \"PABLitO\" we passed all
        # the array \"center\". I think that it is due to the change of type of
        # the input arguments from \"dvector\" to \"darray\".
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
        numpy_t_o_centers = [numpy.array(t_o_center) for t_o_center in \
                             t_o_centers]
        for idx in idxs[0]:
            # Foreground transformation matrix adjoint's dictionary.
            f_t_adj_dict = get_trans_adj(int(keys[idx][0]))
            f_t_dict = get_trans(int(keys[idx][0]))
            numpy_i_center = narray(i_centers[idx])
            t_i_center =  apply_persp_trans(dimension     ,
                                            numpy_i_center,
                                            f_t_dict)[: dimension]

            neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
            (neigh_centers, 
             neigh_indices)  = find_right_neighbours(local_idxs[idx],
                                                     o_ranges[0]    ,
                                                     True)
            coeffs = least_squares(neigh_centers,
                                   numpy_t_o_centers[idx])

            coeffs = [coeff * values_to_multiply[idx] for coeff in coeffs]
            apply_rest_prol_ops(int(keys[idx][1]),
                                neigh_indices    ,
                                coeffs           ,
                                neigh_centers)

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

        octree = self._octree
        py_oct = octree.get_octant(current_octant)
        centers = []
        indices = []
        grid = self._proc_g
        dimension = self._dim
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
        c_c = octree.get_center(current_octant)[: dimension]

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
        narray = numpy.array
        # Lambda function.
        f_n = lambda x, y : find_neighbours(current_octant,
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
            if (neighs):
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
            # consider the indices and centers found (foreground).
            else:
                to_consider = True
                border_center = neighbour_centers(c_c  ,
                                                  codim,
                                                  face_node)

                if (not is_background):
                    numpy_border_center = narray(border_center)
                    t_center =  apply_persp_trans(dimension          ,
                                                  numpy_border_center,
                                                  c_t_dict)[: dimension]
                    check = is_point_inside_polygon(t_center    ,
                                                    t_background)
                    to_consider = (not check)
                    
                if (to_consider):
                    centers.append(border_center)
                    indices.append("outside_bg")

        numpy_centers = numpy.array(centers)

        return (numpy_centers, indices)
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

