# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# A guide to analyzing Python performance:
# http://www.huyng.com/posts/python-performance-analysis/
# ------------------------------------IMPORT------------------------------------
import numbers
import math
import collections
import BaseClass2D
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
        self._n_grids = kwargs["total number of grids"]
        # Number of octants for each grid. It is a list.
        self._oct_f_g = kwargs["octants for grids"]
        # Getting the rank of the current process inside the world communicator
        # and inside the local one.
        self._rank_w = self._comm_w.Get_rank()
        self._rank = self._comm.Get_rank()
        t_points = kwargs.setdefault("transformed points", None)
        self._ref = kwargs["refinement"]
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
                          cs     ,          # Centers
                          es_o_ns,          # Edges or nodes
                          vs     ,          # Values
                          hs     ,          # Edge sizes
                          r_a_n_d = False): # Return also numpy data
        """Function which returns the centers of neighbours, depending on
           for which face we are interested into.

           Arguments:
               cs (tuple or list of tuple) : coordinates of the centers of
                                             the current octree.
               es_o_ns (int between 1 and 2 or list) : numbers indicating if the
                                                       neighbour is from edge or
                                                       node.
               vs (int between 0 and 3 or list) : faces or nodesfor which we are
                                                  interested into knowing the
                                                  neighbour's center.
               hs (list) : size of faces on the boundary for each octant.

           Returns:
               a tuple or a list containing the centers evaluated."""

        centers = cs
        values = vs
        edges_or_nodes = es_o_ns
        h_s = hs
        # Checking if passed arguments are lists or not. If not, we have to do
        # something.
        try:
            len(centers)
            len(values)
            len(edges_or_nodes)
            len(h_s)
        # \"TypeError: object of type 'int' has no len()\", so are no lists but
        # single elements.
        except TypeError:
            t_center, t_value, t_e_o_n = centers, values, edges_or_nodes
            centers, values, edges_or_nodes, h_s = ([] for i in range(0, 4))
            centers.append(t_center)
            values.append(t_value)
            edges_or_nodes.append(t_e_o_n)
            h_s.append(hs)

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
            h = h_s[i]
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

        if (r_a_n_d):
            return (eval_centers, numpy.asarray(eval_centers))

        return eval_centers
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Evaluate boundary conditions.
    def eval_b_c(self   ,
                 centers,
                 f_o_n  ,
                 h_s    ,
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
               h_s (list) : size of faces on the boundary for each octant.
               codim (list) : list of index to point out if is a neighbour of
                              face (\"1\") or node (\"2\").

           Returns:
               boundary_values (int or list) : the evaluated boundary condition
                                               or a list of them.
               c_neighs (tuple or list of tuples) : the centers where evaluate
                                                    the boundary conditions."""

        edges_or_nodes = []
        just_one_neighbour = False
        proc_grid = self._proc_g
        grid = self._proc_g
        c_t_dict = self.get_trans(grid)[0]
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        dimension = self._dim
        solution = utilities.exact_sol
        narray = numpy.array
        nsolution = lambda x : solution(narray(x),
                                        alpha    ,
                                        beta     ,
                                        dim = 2)

        if (codim is None):
            for i in xrange(0, len(centers)):
                # Evaluating boundary conditions for edges.
                edges_or_nodes.append(1)
        else:
            edges_or_nodes = codim
        # Centers neighbours.
        c_neighs = self.neighbour_centers(centers       ,
                                          edges_or_nodes,
                                          f_o_n         ,
                                          h_s)
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

        x_y_s = zip(x_s, y_s)
        boundary_values = nsolution(x_y_s)

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
    def set_bg_b_c(self          ,
                   inter         ,
                   m_octant      ,
                   owners_centers,
                   n_normal_inter,
                   labels        ,
                   multiplier):
        """Method which set boundary condition for an octant owned by the back-
           ground grid.

           Arguments:
                inter (uintptr_t) : pointer to intersection.
                m_octant (int) : masked global index of the octant for which set
                                 the boundary condition.
                owners_centers (list of list) : centers of the owners of the in-
                                                tersection.
                n_normal_inter (numpy array) : normal to the intersection.
                labels (list) : list of 0s or 1s to know if the owners are with
                                the normal inside (0)  or outside (1).
                multiplier (float): coefficient to multiply with the boundary
                                    condition evaluated in this function."""

        octree = self._octree
        n_codim = 1
        dimension = self._dim
        insert_mode = PETSc.InsertMode.ADD_VALUES
        n_axis = numpy.nonzero(n_normal_inter)[0][0]

        if (n_axis):
            if (n_normal_inter[n_axis] == 1):
                n_face = 3
            else:
                n_face = 2
        else:
            if (n_normal_inter[n_axis] == 1):
                n_face = 1
            else:
                n_face = 0

        h = octree.get_area(inter        ,
                            is_ptr = True,
                            is_inter = True)

        n_center = owners_centers[1 - labels[0]][: dimension]

        b_value, \
        c_neigh = self.eval_b_c(n_center,
                                n_face  ,
                                h       ,
                                n_codim)
        # The multiplication for \"(-1 * mult)\" is due to
        # the fact that we have to move the coefficients to
        # the \"rhs\", so we have to change its sign.
        b_value = b_value * -1.0
        self._rhs.setValues(m_octant            ,
                            b_value * multiplier,
                            insert_mode)
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

        #if (m_g_octant == 4613):
        #    print(g_octant)
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
        dimension = self._dim
        octree = self._octree
        n_grids = self._n_grids

        # Code hoisting.
        get_nodes = octree.get_nodes
        get_center = octree.get_center
        get_global_idx = octree.get_global_idx
        get_ghost_global_idx = octree.get_ghost_global_idx
        get_ghost_octant = octree.get_ghost_octant
        find_neighbours = octree.find_neighbours
        check_oct_corners = utilities.check_oct_corners
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

            if ((n_grids > 1) and (is_background)):
                t_foregrounds = numpy.array([self._t_foregrounds])
                # Current transformation matrix's dictionary.
                alpha = self.get_trans(0)[1]
                beta = self.get_trans(0)[2]
                idx_or_oct = neighs[i] if (not ghosts[i]) else \
                             py_ghost_oct
                is_ptr = False if (not ghosts[i]) else \
                         True

                oct_corners, \
                numpy_corners = get_nodes(idx_or_oct,
                                          dimension ,
                                          is_ptr    ,
                                          also_numpy_nodes = True)

                is_n_penalized, \
                n_polygon = check_oct_corners(numpy_corners,
                                              alpha        ,
                                              beta         ,
                                              t_foregrounds)
            if (not is_penalized):
                if (is_n_penalized):
                    # Being the neighbour penalized, it means that it will
                    # be substituted by 4 octants being part of the foreground
                    # grids, so being on the non diagonal part of the grid.
                    # TODO: This is the worst case, not always presents. We
                    #       should find a better way to evaluate the right num-
                    #       ber of neighbours (better case has 3 octants on the
                    #       foreground grids, due to a planar interpolation).
                    # TODO: with least squares method, it should be augmented
                    #       by 8.
                    o_count += 4
                else:
                    if (ghosts[i]):
                        o_count += 1
                    else:
                        d_count += 1
            else:
                if (not is_n_penalized):
                    #if (get_global_idx(octant) == 200):
                    #    print("index " + str(index) + " codim " + str(codim) + " f_o_n " + str(f_o_n))
                    stencil = self._edl.get(key)
                    stencil[s_i] = index
                    s_i += 2

            extra_msg = ""

            msg = "Checked neighbour for "               + \
                  ("edge " if (codim == 1) else "node ") + \
                  str(index)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
        #if (get_global_idx(octant) == 200):
        #    print("index " + str(index) + " codim " + str(codim) + " f_o_n " + str(f_o_n))

        return (d_count,
                o_count,
                s_i    ,
                n_neighbours)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Creates masking system for the octants of the background grid covered by
    # the foreground meshes, and also determines the number of non zero elements
    # to allocate for each row in the system's matrix.
    #@profile
    def create_mask(self):
        """Method which creates the new octants' numerations and initialize non
           zero elements' number for row in the matrix of the system.

           Returns:
               (d_nnz, o_nnz) (tuple) : two lists containting the diagonal
                                        block's and non diagonal block's number
                                        of non zero elements."""

        grid = self._proc_g
        n_oct = self._n_oct
        n_grids = self._n_grids
        octree = self._octree
        nfaces = octree.get_n_faces()
        dimension = self._dim
        is_background = True if (not grid) else False
        if (n_grids > 1):
            t_foregrounds = numpy.array([self._t_foregrounds])
        # Lists containing number of non zero elements for diagonal and non
        # diagonal part of the coefficients matrix, for row.
        d_nnz, o_nnz = ([] for i in range(0, 2))
        h_s = []
        new_oct_count = 0
        # \"range\" gives us a list, \"xrange\" an iterator.
        octants = xrange(0, n_oct)
        # TODO: use a parallel trick:
        # http://stackoverflow.com/questions/5236364/how-to-parallelize-list-comprehension-calculations-in-python
        g_octants = [octree.get_global_idx(octant) for octant in octants]
        py_octs = [octree.get_octant(octant) for octant in octants]
        centers = [octree.get_center(octant)[: dimension] for octant in octants]
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]

        # Code hoisting.
        get_nodes = octree.get_nodes
        get_bound = octree.get_bound
        check_neighbours = self.check_neighbours
        check_oct_corners = utilities.check_oct_corners
        # TODO: try to parallelize this for avoiding data dependencies.
        for octant in octants:
            d_count, o_count = 0, 0
            g_octant = g_octants[octant]
            py_oct = py_octs[octant]
            h = octree.get_area(py_oct       ,
                                is_ptr = True,
                                is_inter = False)
            # Lambda function.
            g_b = lambda x : get_bound(py_oct,
                                       x)
            center = centers[octant]
            # Check to know if an octant is penalized.
            is_penalized = False
            # Background grid.
            if ((n_grids > 1) and (is_background)):
                oct_corners, \
                numpy_corners = get_nodes(octant   ,
                                          dimension,
                                          also_numpy_nodes = True)

                is_penalized, \
                n_polygon = check_oct_corners(numpy_corners,
                                              alpha        ,
                                              beta         ,
                                              t_foregrounds)
            #if (g_octant == 224):
            #    print(is_penalized)
            if (is_penalized):
                self._nln[octant] = -1
                self._p_o_f_g[octant] = n_polygon
                # Moved \"h\" from the \"key\" to the \"stencil\", preferring
                # not to use float into dict keys.
                key_0 = n_polygon + 1 # Foreground grid to which the node be-
                                      # longs to (\"+ 1\" because foreground
                                      # grids starts from 1, globally)
                key_1 = g_octant      # Global index (not yet masked)
                key_2 = 0             # Useless field to pair with foreground keys

                key = (key_0,
                       key_1,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2,
                       key_2)

                # If the octant is covered by the foreground grids, we need to
                # store info of the stencil it belongs to, to push on the rela-
                # tive rows of the matrix, the right indices of the octants of
                # the foreground grid owning the penalized one:
                # first, second and third \"stencil\"'s elements: center of the
                # penalized octant;
                # others \"stencil\"'s elements: global indices and bilinear
                # coefficients to multiply approximation (being in a case of a
                # possible jump of 1 level between elements, we have to store
                # two possible neighbours for each face of the current octant).
                l_stencil = 21 if (dimension == 2) else 31
                stencil = [0, -1] * (l_stencil/2)
                stencil.append(0)
                stencil[0] = h
                for i in xrange(dimension):
                    stencil[i + 1] = center[i]
                # http://www.laurentluce.com/posts/python-dictionary-implementation/
                # http://effbot.org/zone/python-hash.htm
                self._edl.update({key : stencil})
            else:
                self._nln[octant] = new_oct_count
                self._p_o_f_g[octant] = -1
                new_oct_count += 1
                d_count += 1
                h_s.append(h)
            # \"stencil\"'s index.
            s_i = 1 + dimension
            # Number of neighbours (Being the possibility of a jump between oc-
            # tants, we can have a minimum of 4 and a maximum of 8 neighbours on
            # the faces.
            n_neighbours = 0
            # Faces' loop.
            n_faces_loop = nfaces
            if (is_penalized):
                for n_node in xrange(0, nfaces):
                    c_dim = 2
                    d_count, \
                    o_count, \
                    s_i    , \
                    n_neighs = check_neighbours(c_dim                        ,
                                                n_node                       ,
                                                octant                       ,
                                                o_count                      ,
                                                d_count                      ,
                                                s_i                          ,
                                                key if is_penalized else None,
                                                is_penalized                 ,
                                                is_background)
                    n_neighbours = n_neighbours + n_neighs
            for face in xrange(0, nfaces):
                n_face = face
                c_dim = 1
                # Not boundary face.
                if (not g_b(face)):
                    d_count, \
                    o_count, \
                    s_i    , \
                    n_neighs = check_neighbours(c_dim                        ,
                                                n_face                       ,
                                                octant                       ,
                                                o_count                      ,
                                                d_count                      ,
                                                s_i                          ,
                                                key if is_penalized else None,
                                                is_penalized                 ,
                                                is_background)
                    n_neighbours = n_neighbours + n_neighs
                else:
                    if (not is_background):
                        # TODO: with least squares method, it should be augmented
                        #       by 8.
                        # Extern \"ghost\" octant for foreground grids will be
                        # approximated by just its background owner.
                        o_count += 1

            # For the moment, we have to store space in the \"PETSc\" matrix for
            # the octants that will interpolate with the bilinear method (4
            # octants in 2D at maximum) for the vertices of each intersection.
            # And these vertices are equal to the number of neighbours of the
            # current octant (With a gap of one level, we can have as maximum
            # two neighbours for each face), but we do not know a priori if the
            # owners of the nodes will be ghosts (\"o_count\"), or not
            # (\"d_count\").
            # TODO: find a better algorithm to store just the right number of e-
            # lements for \"d_count\" and for \"o_count\".
            # TODO: we try not to consider them using just the ghost stencils of
            #       face neighbours for the borders of foreground grids, because
            #       we have seen that we will not obtain a second order using
            #       the diamond method.
            d_count += (4 * n_neighbours)
            o_count += (4 * n_neighbours)
            if (not is_penalized):
                d_nnz.append(d_count)
                o_nnz.append(o_count)
                self._centers_not_penalized.append(center)

        self.new_spread_new_background_numeration(is_background)

        sizes = self.find_sizes()
        # Max value that can be inserted into \"d_nnz\". It is equal to the lo-
        # cal dimension of the matrix in the current octant.
        max_d_nnz = sizes[0]
        # Max value that can be inserted into \"o_nnz\". It is equal to the to-
        # tal dimension of the matrix minus the local one of the current octant.
        max_o_nnz = sizes[1] - sizes[0]
        # A \"numpy\" version of \"d_nnz\" list.
        n_d_nnz = numpy.array(d_nnz)
        # A \"numpy\" version of \"o_nnz\" list.
        n_o_nnz = numpy.array(o_nnz)
        # Imposing maximum value of numbers contained in \"d_nnz\" and \"o_nnz\"
        # because, if not done, the values could exceed the \"sizes\" of the ma-
        # trix, especially for little matrices.
        n_d_nnz[n_d_nnz > max_d_nnz] = max_d_nnz
        n_o_nnz[n_o_nnz > max_o_nnz] = max_o_nnz
        d_nnz = n_d_nnz.tolist()
        o_nnz = n_o_nnz.tolist()

        msg = "Created mask"
        self.log_msg(msg   ,
                     "info")

        n_h_s = numpy.array(h_s)
        return (d_nnz,
                o_nnz,
                n_h_s)
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
        if (is_background):
            to_send = self._p_o_f_g
        to_receive = self._g_p_o_f_g

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

    def get_interface_coefficients_1_order(self            ,
                                           inter           ,
                                           dimension       ,
                                           nodes_inter     ,
                                           owners_centers  ,
                                           grid            ,
                                           use_inter = True,
                                           h_given = 0     ,
                                           n_axis_given = 0,
                                           n_value_given = 0):
        octree = self._octree
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        is_bound_inter = True
        n_axis = n_axis_given
        n_value = n_value_given
        n_normal_inter = numpy.zeros((3, ),
                                     dtype = numpy.int64)
        n_normal_inter[n_axis] = n_value
        h = h_given
        if (use_inter):
            is_bound_inter = octree.get_bound(inter,
                                              0    ,
                                              True)
            # Normal to the intersection, and its numpy version.
            normal_inter, \
            n_normal_inter = octree.get_normal(inter,
                                               True) # We want also a \"numpy\"
                                                     # version
            n_axis = numpy.nonzero(n_normal_inter)[0][0]
            n_value = n_normal_inter[n_axis]
            # evaluating length of the intersection.
            h = octree.get_area(inter        ,
                                is_ptr = True,
                                is_inter = True)
        h_inv = (1.0 / h)

        d_nodes_x    , \
        d_nodes_y    , \
        c_inter      , \
        d_o_centers_x, \
        d_o_centers_y = self.get_interface_distances(dimension     ,
                                                     h             ,
                                                     nodes_inter   ,
                                                     owners_centers,
                                                     is_bound_inter,
                                                     n_normal_inter)

        coeff_in_grad_x = (1.0 / d_o_centers_x) if (d_o_centers_x != 0.0) else \
                          0.0
        coeff_in_grad_y = (1.0 / d_o_centers_y) if (d_o_centers_y != 0.0) else \
                          0.0
        coeff_out_grad_x = -1.0 * coeff_in_grad_x
        coeff_out_grad_y = -1.0 * coeff_in_grad_y

        grad_transf = utilities.jacobian_bil_mapping(numpy.array(c_inter),
                                                     alpha               ,
                                                     beta                ,
                                                     dim = 2)
        grad_transf_inv = numpy.linalg.inv(grad_transf)
        grad_transf_det = numpy.linalg.det(grad_transf)
        grad_transf_det_inv = (1.0 / grad_transf_det)
        cofactors = (grad_transf_inv * grad_transf_det).T

        coeffs_trans = numpy.dot(grad_transf_inv, cofactors)

        coeff_trans_x = coeffs_trans[0][1] if (n_axis) else \
                        coeffs_trans[0][0]
        coeff_trans_y = coeffs_trans[1][1] if (n_axis) else \
                        coeffs_trans[1][0]

        n_coeffs_grad_x = numpy.array([coeff_in_grad_x,
                                       coeff_out_grad_x])
        n_coeffs_grad_y = numpy.array([coeff_in_grad_y,
                                       coeff_out_grad_y])

        n_coeffs_grad_x = n_coeffs_grad_x * (h                     * \
                                             coeff_trans_x         * \
                                             n_value)
        n_coeffs_grad_y = n_coeffs_grad_y * (h                     * \
                                             coeff_trans_y         * \
                                             n_value)
        n_coeffs = n_coeffs_grad_x + n_coeffs_grad_y

        return n_coeffs

    def get_gradient_coefficients(self             ,
                                  inter            ,  # pointer to the intersection
                                  dimension        ,  # 2D/3D
                                  nodes_inter      ,  # Coordinates of the nodes
                                                      # of the intersection
                                  owners_centers   ,  # Centers of the owners of
                                                      # the intersection
                                  l_s_coeffs       ,  # Least square coefficients
                                  use_inter = True ,
                                  h_given = 0      ,  # If \"use_inter\" is False,
                                                      # then we use \"h_given\"
                                                      # to evaluate the coeffs
                                  n_axis_given = 0 ,  # Same explication as for
                                  n_value_given = 0,  # \"h_given\".
                                  grid = -1):
        octree = self._octree
        if (grid == -1):
            grid = self._proc_g
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        is_bound_inter = False
        n_axis = n_axis_given
        n_value = n_value_given
        n_normal_inter = numpy.zeros((3, ),
                                     dtype = numpy.int64)
        n_normal_inter[n_axis] = n_value
        h = h_given
        if (use_inter):
            is_bound_inter = octree.get_bound(inter,
                                              0    ,
                                              True)
            # Normal to the intersection, and its numpy version.
            normal_inter, \
            n_normal_inter = octree.get_normal(inter,
                                               True) # We want also a \"numpy\"
                                                     # version
            n_axis = numpy.nonzero(n_normal_inter)[0][0]
            n_value = n_normal_inter[n_axis]
            # evaluating length of the intersection.
            h = octree.get_area(inter        ,
                                is_ptr = True,
                                is_inter = True)
        h_inv = (1.0 / h)

        d_nodes_x    , \
        d_nodes_y    , \
        c_inter      , \
        d_o_centers_x, \
        d_o_centers_y = self.get_interface_distances(dimension     ,
                                                     h             ,
                                                     nodes_inter   ,
                                                     owners_centers,
                                                     is_bound_inter,
                                                     n_normal_inter)

        #den = (d_o_centers_x * d_nodes_y) - \
        #      (d_o_centers_y * d_nodes_x)
        #den_inv = (1.0 / den)

        #coeff_in_grad_x = d_nodes_y
        #coeff_in_grad_y = -1.0 * d_nodes_x
        #coeff_out_grad_x = -1.0 * coeff_in_grad_x
        #coeff_out_grad_y = -1.0 * coeff_in_grad_y
        #coeff_node_1_grad_x = -1.0 * d_o_centers_y
        #coeff_node_1_grad_y = d_o_centers_x
        #coeff_node_0_grad_x = -1.0 * coeff_node_1_grad_x
        #coeff_node_0_grad_y = -1.0 * coeff_node_1_grad_y

        grad_transf = utilities.jacobian_bil_mapping(numpy.array(c_inter),
                                                     alpha               ,
                                                     beta                ,
                                                     dim = 2)
        grad_transf_inv = numpy.linalg.inv(grad_transf)
        grad_transf_det = numpy.linalg.det(grad_transf)
        grad_transf_det_inv = (1.0 / grad_transf_det)
        cofactors = (grad_transf_inv * grad_transf_det).T

        coeffs_trans = numpy.dot(grad_transf_inv, cofactors)

        coeff_trans_x = coeffs_trans[0][1] if (n_axis) else \
                        coeffs_trans[0][0]
        coeff_trans_y = coeffs_trans[1][1] if (n_axis) else \
                        coeffs_trans[1][0]

        return (coeff_trans_x * h * n_value,
                coeff_trans_y * h * n_value)

        #n_coeffs_grad_x = numpy.array([coeff_in_grad_x    ,
        #                               coeff_out_grad_x   ,
        #                               coeff_node_1_grad_x,
        #                               coeff_node_0_grad_x])
        #n_coeffs_grad_y = numpy.array([coeff_in_grad_y    ,
        #                               coeff_out_grad_y   ,
        #                               coeff_node_1_grad_y,
        #                               coeff_node_0_grad_y])

        #n_coeffs_grad_x = n_coeffs_grad_x * (den_inv               * \
        #                                     h                     * \
        #                                     coeff_trans_x         * \
        #                                     n_value)
        #n_coeffs_grad_y = n_coeffs_grad_y * (den_inv               * \
        #                                     h                     * \
        #                                     coeff_trans_y         * \
        #                                     n_value)
        #n_coeffs = n_coeffs_grad_x + n_coeffs_grad_y

        #mult_node_1 = 1.0
        #mult_node_0 = 1.0
        ## If the nodes are not on the background boundary, we have evaluated
        ## bilinear interpolation to interpolate the nodes, indeed. On the coun-
        ## trary, being on the background border, it will have the exact value of
        ## the solution on that node, so there will not be the interpolation
        ## coefficients..
        #if (l_s_coeffs[1].size):
        #    mult_node_1 = l_s_coeffs[1]
        #if (l_s_coeffs[0].size):
        #    mult_node_0 = l_s_coeffs[0]

        #coeffs_node_1 = mult_node_1 * n_coeffs[2]
        #coeffs_node_0 = mult_node_0 * n_coeffs[3]

        #return (n_coeffs     ,
        #        coeffs_node_1,
        #        coeffs_node_0)


    def get_interface_coefficients(self             ,
                                   inter            ,  # pointer to the intersection
                                   dimension        ,  # 2D/3D
                                   nodes_inter      ,  # Coordinates of the nodes
                                                       # of the intersection
                                   owners_centers   ,  # Centers of the owners of
                                                       # the intersection
                                   l_s_coeffs       ,  # Least square coefficients
                                   use_inter = True ,
                                   h_given = 0      ,  # If \"use_inter\" is False,
                                                       # then we use \"h_given\"
                                                       # to evaluate the coeffs
                                   n_axis_given = 0 ,  # Same explication as for
                                   n_value_given = 0,  # \"h_given\".
                                   grid = -1):
        octree = self._octree
        if (grid == -1):
            grid = self._proc_g
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        is_bound_inter = False
        n_axis = n_axis_given
        n_value = n_value_given
        n_normal_inter = numpy.zeros((3, ),
                                     dtype = numpy.int64)
        n_normal_inter[n_axis] = n_value
        h = h_given
        if (use_inter):
            is_bound_inter = octree.get_bound(inter,
                                              0    ,
                                              True)
            # Normal to the intersection, and its numpy version.
            normal_inter, \
            n_normal_inter = octree.get_normal(inter,
                                               True) # We want also a \"numpy\"
                                                     # version
            n_axis = numpy.nonzero(n_normal_inter)[0][0]
            n_value = n_normal_inter[n_axis]
            # evaluating length of the intersection.
            h = octree.get_area(inter        ,
                                is_ptr = True,
                                is_inter = True)
        h_inv = (1.0 / h)

        d_nodes_x    , \
        d_nodes_y    , \
        c_inter      , \
        d_o_centers_x, \
        d_o_centers_y = self.get_interface_distances(dimension     ,
                                                     h             ,
                                                     nodes_inter   ,
                                                     owners_centers,
                                                     is_bound_inter,
                                                     n_normal_inter)

        den = (d_o_centers_x * d_nodes_y) - \
              (d_o_centers_y * d_nodes_x)
        den_inv = (1.0 / den)

        coeff_in_grad_x = d_nodes_y
        coeff_in_grad_y = -1.0 * d_nodes_x
        coeff_out_grad_x = -1.0 * coeff_in_grad_x
        coeff_out_grad_y = -1.0 * coeff_in_grad_y
        coeff_node_1_grad_x = -1.0 * d_o_centers_y
        coeff_node_1_grad_y = d_o_centers_x
        coeff_node_0_grad_x = -1.0 * coeff_node_1_grad_x
        coeff_node_0_grad_y = -1.0 * coeff_node_1_grad_y

        grad_transf = utilities.jacobian_bil_mapping(numpy.array(c_inter),
                                                     alpha               ,
                                                     beta                ,
                                                     dim = 2)
        grad_transf_inv = numpy.linalg.inv(grad_transf)
        grad_transf_det = numpy.linalg.det(grad_transf)
        grad_transf_det_inv = (1.0 / grad_transf_det)
        cofactors = (grad_transf_inv * grad_transf_det).T

        coeffs_trans = numpy.dot(grad_transf_inv, cofactors)

        coeff_trans_x = coeffs_trans[0][1] if (n_axis) else \
                        coeffs_trans[0][0]
        coeff_trans_y = coeffs_trans[1][1] if (n_axis) else \
                        coeffs_trans[1][0]

        n_coeffs_grad_x = numpy.array([coeff_in_grad_x    ,
                                       coeff_out_grad_x   ,
                                       coeff_node_1_grad_x,
                                       coeff_node_0_grad_x])
        n_coeffs_grad_y = numpy.array([coeff_in_grad_y    ,
                                       coeff_out_grad_y   ,
                                       coeff_node_1_grad_y,
                                       coeff_node_0_grad_y])

        n_coeffs_grad_x = n_coeffs_grad_x * (den_inv               * \
                                             h                     * \
                                             coeff_trans_x         * \
                                             n_value)
        n_coeffs_grad_y = n_coeffs_grad_y * (den_inv               * \
                                             h                     * \
                                             coeff_trans_y         * \
                                             n_value)
        n_coeffs = n_coeffs_grad_x + n_coeffs_grad_y

        mult_node_1 = 1.0
        mult_node_0 = 1.0
        # If the nodes are not on the background boundary, we have evaluated
        # bilinear interpolation to interpolate the nodes, indeed. On the coun-
        # trary, being on the background border, it will have the exact value of
        # the solution on that node, so there will not be the interpolation
        # coefficients..
        if (l_s_coeffs[1].size):
            mult_node_1 = l_s_coeffs[1]
        if (l_s_coeffs[0].size):
            mult_node_0 = l_s_coeffs[0]

        coeffs_node_1 = mult_node_1 * n_coeffs[2]
        coeffs_node_0 = mult_node_0 * n_coeffs[3]

        return (n_coeffs     ,
                coeffs_node_1,
                coeffs_node_0)

    def get_interface_distances(self          ,
                                dimension     ,
                                inter_size    ,
                                nodes_inter   ,
                                owners_centers,
                                is_bound_inter,
                                n_normal_inter):
        h = inter_size
        d_nodes_x = 0.0
        d_nodes_y = 0.0
        d_o_centers_x = 0.0
        d_o_centers_y = 0.0
        # Normal axis.
        n_axis = numpy.nonzero(n_normal_inter)[0][0]
        # Normal value.
        n_value = n_normal_inter[n_axis]

        if (n_axis):
            d_nodes_x = h
        else:
            d_nodes_y = h
        # Center of the intersection.
        c_inter = ((nodes_inter[1][0] + nodes_inter[0][0]) / 2.0,
                   (nodes_inter[1][1] + nodes_inter[0][1]) / 2.0)
        if (is_bound_inter):
            mult = 1.0 if (n_value > 0) else -1.0
            # Normal parallel to y-axis.
            if (n_axis):
                # Distance between y of center of the octant owner of
                # the intersection and the extern boundary.
                d_o_centers_y = mult * h
            # Normal parallel to x-axis.
            else:
                # Distance between x of center of the octant owner of
                # the intersection and the extern boundary.
                d_o_centers_x = mult * h
        else:
            # Distance between xs of centers of the octants partaging
            # the intersection.
            #d_o_centers_x = numpy.absolute(owners_centers[1][0] - \
            #                               owners_centers[0][0])
            d_o_centers_x = owners_centers[0][0] - owners_centers[1][0]
            # Distance between ys of centers of the octants partaging
            # the intersection.
            #d_o_centers_y = numpy.absolute(owners_centers[1][1] - \
            #                               owners_centers[0][1])
            d_o_centers_y = owners_centers[0][1] - owners_centers[1][1]

        return (d_nodes_x    ,
                d_nodes_y    ,
                c_inter      ,
                d_o_centers_x,
                d_o_centers_y)

    # TODO: change the name of this function, I really do not like it!
    def get_owners_nodes_inter(self              ,
                               inter             ,
                               l_owners_inter    , # USELESS, because is never
                                                   # called
                               o_ghost           , # Owner ghost; if it is
                                                   # \"None\", there is no ow-
                                                   # ner ghost but, if it is
                                                   # \"0\" or \"1\", it means
                                                   # that is respectively the
                                                   # owner with the inner nor-
                                                   # mal or the one with the
                                                   # outer one.
                               also_nodes = False, # Return also the nodes of
                                                   # the intersection, not
                                                   # just their owners
                               r_a_n_d = False):
        octree = self._octree
        tot_oct = self._tot_oct
        dimension = self._dim
        grid = self._proc_g
        n_grids = self._n_grids
        finer_o_inter = octree.get_finer(inter)
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        # Number of nodes on the foreground boundary.
        n_nodes_on_f_b = 0
        # Node on the foreground boundary.
        node_on_f_b = 0
        are_nodes_on_f_b = [False, False]
        # Index finer owner intersection.
        i_finer_o_inter = octree.get_owners(inter)[finer_o_inter]
        t_background = numpy.array([self._t_background])
        if (n_grids > 1):
            t_foreground = numpy.array([self._t_foregrounds[grid - 1]])
        n_nodes = 2 if (dimension == 2) else 4
        nodes = octree.get_nodes(inter        ,
                                 dimension    ,
                                 is_ptr = True,
                                 is_inter = True)[: n_nodes]
        is_point_on_lines = utilities.is_point_on_lines
        apply_bil_mapping = utilities.apply_bil_mapping
        # Is on background boundary.
        is_on_b_boundary = lambda x : is_point_on_lines(x,
                                                        t_background)
        if (n_grids > 1):
            is_on_f_boundary = lambda x : is_point_on_lines(x,
                                                            t_foreground)
        # Local indices of the octants owners of the nodes of the
        # intersection.
        l_owners = [0] * n_nodes
        #\"Numpy\" transformed node.
        n_t_node = numpy.zeros(shape = (1, 3), dtype = numpy.float64)
        for i in xrange(0, n_nodes):
            on_f_boundary = False # CHANGE \"on_f_boundary = False\" with
                                  # \"on_b_boundary = False\", otherwise we get
                                  # a repetion 12 lines more down here.
            node = (nodes[i][0], nodes[i][1], nodes[i][2])
            n_node = numpy.array([node])
            apply_bil_mapping(n_node  ,
                              alpha   ,
                              beta    ,
                              n_t_node,
                              dim = 2)
            # TODO: use a multiprocess trick to check \"on_b_boundary\" and
            #       \"on_f_boundary\" simultaneously, because there is no pos-
            #       sible race condition.
            on_b_boundary = is_on_b_boundary(n_t_node)
            on_f_boundary = False
            if (grid and (not on_b_boundary)):
                on_f_boundary = is_on_f_boundary(n_t_node)

            if (on_b_boundary):
                l_owner = "b_boundary"
            else:
                if (on_f_boundary):
                    if (not n_nodes_on_f_b):
                        # If both the nodes of the intersection are on the fore-
                        # ground boundary, there is no more need to know which
                        # of the two rely on it.
                        node_on_f_b = i
                    n_nodes_on_f_b += 1
                    are_nodes_on_f_b[i] = True
                # We will not have a case of a ghost owner, because the current
                # function is runned only if the intersection is owned by the
                # current process.
                l_owner = i_finer_o_inter
            l_owners[i] = l_owner

        if (also_nodes):
            if (r_a_n_d):
                return (l_owners            ,
                        nodes               ,
                        # Here \"array\" or \"asarray\" is the same thing,
                        # a copy is always done.
                        numpy.asarray(nodes),
                        n_nodes_on_f_b      ,
                        node_on_f_b         ,
                        are_nodes_on_f_b)

            return (l_owners      ,
                    nodes         ,
                    n_nodes_on_f_b,
                    node_on_f_b   ,
                    are_nodes_on_f_b)

        return (l_owners      ,
                n_nodes_on_f_b,
                node_on_f_b   ,
                are_nodes_on_f_b)

    # --------------------------------------------------------------------------
    # Initialize the monolithic matrix.
    def init_mat(self,
                 (d_nnz, o_nnz)):
        """Method which initialize the monolithic matrix of the system.

           Arguments:
               (d_nnz, o_nnz) (tuple) : two lists containting the diagonal and
                                            non diagonal block's number of non
                                            zero elements, obtained by function
                                            \"create_mask\"."""

        comm_w = self._comm_w

        sizes = self.find_sizes()

        self._b_mat = PETSc.Mat().createAIJ(size = (sizes, sizes),
                                            nnz = (d_nnz, o_nnz) ,
                                            comm = comm_w)
        # If an element is being allocated in a place not preallocate, then
        # the program will stop.
        self._b_mat.setOption(self._b_mat.Option.NEW_NONZERO_ALLOCATION_ERR,
                              False)
        msg = "Initialized monolithic matrix"
        extra_msg = "with sizes \"" + str(self._b_mat.getSizes()) + \
                    "\" and type \"" + str(self._b_mat.getType()) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Fill the diagonal matrices of the block matrix.
    def fill_mat_and_rhs(self):
        """Method which fill the diagonal parts of the monolithic matrix of the
           system."""

        f_bound = self._f_bound
        grid = self._proc_g
        n_grids = self._n_grids

        # Ghosts' deplacement.
        g_d = 0
        for i in xrange(0, grid):
            g_d = g_d + self._oct_f_g[i]

        comm_w = self._comm_w
        rank_w = self._rank_w
        octree = self._octree
        tot_oct = self._tot_oct
        is_background = False if (grid) else True
        if (n_grids > 1):
            t_foregrounds = numpy.array(self._t_foregrounds)

        n_oct = self._n_oct
        nfaces = octree.get_n_faces()
        ninters = octree.get_num_intersections()
        dimension = self._dim

        o_ranges = self.get_ranges()

        # Current transformation matrix's dictionary.
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]

        # Code hoisting.
        mask_octant = self.mask_octant
        get_octant = octree.get_octant
        get_ghost_octant = octree.get_ghost_octant
        get_center = octree.get_center
        get_nodes = octree.get_nodes
        get_intersection = octree.get_intersection
        get_bound = octree.get_bound
        check_oct_corners = utilities.check_oct_corners
        get_owners_normals_inter = self.get_owners_normals_inter
        get_is_ghost = octree.get_is_ghost
        # Interpolation coefficients
        inter_coeffs = self.inter_coeffs
        get_owners_nodes_inter = self.get_owners_nodes_inter
        new_find_right_neighbours = self.new_find_right_neighbours
        set_bg_b_c = self.set_bg_b_c
        fill_rhs = self.fill_rhs
        fill_mat = self.fill_mat
        bil_coeffs = utilities.bil_coeffs
        get_finer = octree.get_finer
        get_intersection_local_rings = octree.get_intersection_local_rings
        # Lambda functions.
        g_n = lambda x : get_nodes(x               ,
                                   dimension       ,
                                   is_ptr = True   ,
                                   is_inter = False,
                                   also_numpy_nodes = True)
        f_r_n = lambda x : new_find_right_neighbours(x[0]       ,
                                                     o_ranges[0],
                                                     x[1]       ,
                                                     True       ,
                                                     x[2]       ,
                                                     x[3]       ,
                                                     True)

        b_c = lambda x: bil_coeffs(x[0],
                                   x[1])
        least_squares = lambda x: utilities.least_squares(x[0],
                                                          x[1])

        self._f_nodes = []
        self._f_nodes_exact = []
        self._f_nodes_on_inter_borders = []
        self._f_nodes_exact_on_inter_borders = []
        self._f_on_borders_exact = []
        self._f_on_borders = []
        self._h_s_inter = []
        self._h_s_inter_on_board = []

        for i in xrange(0, ninters):
            # Rows indices for the \"PETSc\" matrix.
            r_indices = []
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
            # Choosing if the current intersection is owned by the current pro-
            # cess. If yes, do everything; if not (ghost case), do nothing be-
            # cause everything will be done by the other process, real owner of
            # the intersection.
            finer_o_inter = get_finer(inter)
            cur_proc_owner = True
            if (is_ghost_inter):
                if (finer_o_inter):
                    cur_proc_owner = False
            if (cur_proc_owner):
                # Global indices of owners inner/outer normal of the intersection
                # (is a list, and the first element is the one with the inner nor-
                # mal), followed by local indices of owners (needed by functions
                # \"get_octant\" and \"get_ghost_octant\"), and an integer to know
                # what owner is ghost.
                #
                # Here, global means global in the current octree. To have global
                # for the totality of the octrees, we have to add \"g_d\".
                #
                # First owner will be the one with the inner normal.
                #
                # Owner ghost equal to 1 means that is ghost the one with the outer
                # normal, if is 0 is ghost the one with the inner.
                g_o_norms_inter, \
                l_o_norms_inter, \
                o_ghost = get_owners_normals_inter(inter,
                                                   is_ghost_inter)
                # List containing 0 or 1 to indicate inner normal or outer normal.
                labels = []
                # Masked global indices of owners inner/outer normal of the inter-
                # section. REMEMBER: \"octree.get_global_idx(octant) + gd\" ==
                #                    \"o_ranges[0] + octant\".
                # TODO: use \"multiprocessing\" shared memory to map function on
                #       local threads.
                m_g_o_norms_inter = map(mask_octant,
                                        [(g_o_norm_inter + g_d) for g_o_norm_inter \
                                         in g_o_norms_inter])
                # Number of intersection's owners (both in 2D and 3D, it will be
                # always equal to 2).
                n_i_owners = 2
                n_polygon = None
                # Looping on the owners of the intersection.
                for j in xrange(0, n_i_owners):
                    # Here, means that the owner is ghost.
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
                    # If an intersection owner is penalized (it should be just for
                    # background grid)...
                    if (m_g_octant == -1):
                        n_polygon = self._g_p_o_f_g[g_o_norms_inter[j]]
                    # ...else...
                    else:
                        r_indices.append(m_g_octant)
                        if (is_bound_inter):
                            # Normal always directed outside the domain.
                            labels.append(1)
                        else:
                            labels.append(j)
                if (is_bound_inter):
                    # Being a boundary intersection, owner is the same.
                    del r_indices[-1]
                # If the owners of the intersection are not both penalized (row in-
                # dices are not empty).
                if (r_indices):
                    # Local indices of the octants owners of the nodes of the
                    # intersection (needed for \"find_right_neighbours\"
                    # function).
                    #
                    # The coordinates of the nodes are given by \"nodes_inter\".
                    #
                    # The coordinates of the nodes but in a \"numpy\" array are
                    # into \"n_nodes_inter\".
                    #
                    # How many nodes of the current intersection are on the
                    # foreground boundary (for the foreground grids).
                    #
                    # index of the node in \"nodes_inter\" and \"n_nodes_inter\"
                    # which give us the node on the foregound boundaries (this
                    # field is useless if \"n_nodes_on_f_b\" is equal to 2.
                    l_o_nodes_inter, \
                    nodes_inter    , \
                    n_nodes_inter  , \
                    n_nodes_on_f_b , \
                    node_on_f_b    , \
                    are_nodes_on_f_b  = get_owners_nodes_inter(inter            ,
                                                               l_o_norms_inter  ,
                                                               o_ghost          ,
                                                               # Return also coor-
                                                               # dinates of the no-
                                                               # des, and not just
                                                               # local indices of
                                                               # the owners.
                                                               also_nodes = True,
                                                               # Return also \"num-
                                                               # py\" data.
                                                               r_a_n_d = True)
                    # Getting the rings of neighbours for each node of the in-
                    # tersection, excluding the owner of the intersection itself
                    # that will be added later in the other function
                    # \"new_find_right_neighbours\".
                    rings = get_intersection_local_rings(inter)
                    # Neighbour centers neighbours indices: it is a list of tuple,
                    # and in each tuple are contained the lists of centers and in-
                    # dices of each local owner of the nodes.
                    # TODO: use \"multiprocessing\" shared memory to map function on
                    #       local threads.
                    n_cs_n_is = map(f_r_n,
                                    zip(l_o_nodes_inter,
                                        rings          ,
                                        n_nodes_inter  ,
                                        are_nodes_on_f_b))
                    # Least square coefficients.
                    # TODO: use \"multiprocessing\" shared memory to map function on
                    #       local threads.
                    l_s_coeffs = map(b_c,
                                     zip([pair[0] for pair in n_cs_n_is]     ,
                                         [n_node for n_node in n_nodes_inter]))

                    self.compute_function_on_nodes(inter        ,
                                                   n_nodes_inter,
                                                   n_cs_n_is    ,
                                                   l_s_coeffs   ,
                                                   are_nodes_on_f_b)

                    n_coeffs     , \
                    coeffs_node_1, \
                    coeffs_node_0  =  self.get_interface_coefficients(inter         ,
                                                                      dimension     ,
                                                                      nodes_inter   ,
                                                                      owners_centers,
                                                                      l_s_coeffs)

                    coeffs_nodes = (coeffs_node_0,
                                    coeffs_node_1)

                    fill_rhs((l_s_coeffs[0].size,
                              l_s_coeffs[1].size),
                             labels              ,
                             coeffs_nodes        ,
                             r_indices           ,
                             n_nodes_inter)

                    fill_mat(inter            ,
                             owners_centers   ,
                             n_cs_n_is        ,
                             r_indices        ,
                             labels           ,
                             g_o_norms_inter  ,
                             m_g_o_norms_inter,
                             coeffs_nodes     ,
                             n_polygon        ,
                             n_coeffs         ,
                             nodes_inter      ,
                             n_nodes_on_f_b   ,
                             node_on_f_b      ,
                             rings)

        # We have inserted argument \"assembly\" equal to
        # \"PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY\" because the final assembly
        # will be done after inserting the prolongation and restriction blocks.
        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY)
        self.assembly_petsc_struct("rhs")

        mat_sizes = self._b_mat.getSizes()
        mat_type = self._b_mat.getType()

        msg = "Filled diagonal parts of the monolithic matrix"
        extra_msg = "with sizes \"" + str(mat_sizes) + "\" and type \"" + \
                    str(mat_type) + "\""
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
    def reset_partially_array(self,
                              array_to_reset = "sol"):
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
        if (array_to_reset == "sol"):
            array_name = "solution"
            to_reset = self._sol
        elif (array_to_reset == "res"):
            array_name = "residual"
            to_reset = self._residual
        # Resetted array.
        res_arr = self.init_array("resetted partially " + \
                                  str(array_name),
                                  petsc_size = False)

        for i in ids_octree_contained:
            arr_index = self.mask_octant(i)
            if (arr_index != -1):
                arr_value = to_reset.getValue(arr_index)
                res_arr.setValue(i, arr_value)

        res_arr.assemblyBegin()
        res_arr.assemblyEnd()

        msg = "Resetted partially " + \
              str(array_name)
        self.log_msg(msg   ,
                     "info")

        return res_arr
    # --------------------------------------------------------------------------

    def add_rhs(self,
                numpy_array):
        self.add_array(self._rhs        ,
                       numpy_array      ,
                       "right hand side",
                       True)
        msg = "Added array to \"rhs\""
        self.log_msg(msg,
                     "info")

    def add_array(self             ,
                  vec              ,
                  array_to_add     ,
                  a_name = ""      ,
                  petsc_size = True):
        if (not petsc_size):
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

        if (not petsc_size):
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

        self._rhs = self.init_array("right hand side",
                                    True             ,
                                    numpy_array)

        msg = "Initialized \"rhs\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Initializes \"sol\".
    def init_sol(self,
                 numpy_array = None):
        """Method which initializes the solution."""

        self._sol = self.init_array("solution",
                                    True      ,
                                    numpy_array)

        msg = "Initialized \"solution\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def evaluate_residual_norms(self             ,
                                exact_sol        ,
                                h_s              ,
                                petsc_size = True,
                                l2 = False       ,
                                r_n_d = False):
        if (not petsc_size):
            n_oct = self._n_oct
            tot_oct = self._tot_oct
            sizes = (n_oct, tot_oct)
        else:
            sizes = self.find_sizes()
        # Temporary PETSc exact solution.
        t_e_sol = self.init_array("exact solution",
                                  petsc_size      ,
                                  array = exact_sol)
        # Temporary PETSc array.
        self._residual = self.init_array("residual",
                                         petsc_size,
                                         array = None)
        # \"self._residual\" = \"self._b_mat\" * \"t_e_sol\"
        self._b_mat.mult(t_e_sol,
                         self._residual)
        # \"self._residual\" = \"self._residual\" - \"self._rhs\"
        self._residual.axpy(-1.0,
                            self._rhs)
        norm_inf = numpy.linalg.norm(self._residual.getArray(),
                                     # Type of norm we want to evaluate.
                                     numpy.inf)
        norm_L2 = numpy.linalg.norm(self._residual.getArray() * h_s,
                                    2)

        msg = "Evaluated residuals"
        extra_msg = "with (norm_inf, norm_L2) = " + str((norm_inf, norm_L2))

        if (r_n_d):
            return (numpy.array([norm_inf]),
                    numpy.array([norm_L2]))

        return (norm_inf, norm_L2)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Solving...
    # https://pypi.python.org/pypi/memory_profiler
    # https://www.pluralsight.com/blog/tutorials/how-to-profile-memory-usage-in-python
    #comm_rank = MPI.COMM_WORLD.Get_rank()
    #mem_log_file = "./log/" + str(comm_rank) + "_mem.log"
    #mem_fp=open(mem_log_file,'w+')
    #from memory_profiler import profile
    #@profile(stream=mem_fp)
    def solve(self):
        #print(self._masked_oct_bg_g)
        """Method which solves the system."""
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        pc = PETSc.PC()
        ksp.create(self._comm_w)
        ksp.setOperators(self._b_mat,
                         None)

        # Setting tolerances.
	# 1.0e-06 with level 9,9 is still ok for convergence
        #tol = 1.0e-06
        tol = 1.0e-09
        ksp.setTolerances(rtol = tol            ,
                          atol = tol            ,
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
        ksp.setType("gmres")
        ksp.setFromOptions()
        ksp.setInitialGuessNonzero(True)
        # View the matrix.
        #self._b_mat.view()
        #if (PETSc.COMM_WORLD.getSize() == 1):
        #    # File viewer name
        #    fv_name = "./data/matrix_bin_1_core.dat"
        #else:
        #    fv_name = "./data/matrix_bin.dat"
        # https://fenicsproject.org/qa/7811/export-petsc-matrix-and-vector-in-file
        # https://pythonhosted.org/petsc4py/apiref/petsc4py.PETSc.Viewer-class.html
        # New binary viewer:
        #nv = PETSc.Viewer().createBinary(name = fv_name                    ,
        #                                 mode = PETSc.Viewer().Mode().WRITE,
        #                                 comm = PETSc.COMM_WORLD)
        #self._b_mat.view(nv)
        # Solve the system.
        ksp.solve(self._rhs,
                  self._sol)
        # How many iterations are done.
        it_number = ksp.getIterationNumber()
        print(ksp.getConvergedReason())

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
        # Penalized octants foreground grids.
        self._p_o_f_g = numpy.empty(n_oct,
                                    dtype = numpy.int64)
        # New global numeration.
        self._ngn = numpy.empty(N_oct_bg_g,
                                dtype = numpy.int64)
        # Global penalized octants foreground grids.
        self._g_p_o_f_g = numpy.empty(N_oct_bg_g,
                                      dtype = numpy.int64)
        self._centers_not_penalized = []

        # Numpy edl.
        self._n_edl = None
        # Numpy edg. The \"self._n_edg\" will contains the excahnged data
        # between grids of different levels.
        self._n_edg = None
        # TODO: check to see if is better to use int64 or uint64.
        if not is_background:
            self._d_type_s = numpy.dtype('(1, 13)i8, (1, 21)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 13)i8, (1, 31)f8')
            blocks_length_s = [13, 21] if (dimension == 2) else [13, 31]
            blocks_displacement_s = [0, 104]
            mpi_datatypes = [MPI.INT64_T,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 13)i8, (1, 21)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 13)i8, (1, 31)f8')
            blocks_length_r = [13, 21] if (dimension == 2) else [13, 31]
            blocks_displacement_r = [0, 104]
        else:
            self._d_type_s = numpy.dtype('(1, 13)i8, (1, 21)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 13)i8, (1, 31)f8')
            blocks_length_s = [13, 21] if (dimension == 2) else [13, 31]
            blocks_displacement_s = [0, 104]
            mpi_datatypes = [MPI.INT64_T,
                             MPI.DOUBLE]
            self._d_type_r = numpy.dtype('(1, 13)i8, (1, 21)f8') if \
                             (dimension == 2) else                 \
                             numpy.dtype('(1, 13)i8, (1,31)f8')
            blocks_length_r = [13, 21] if (dimension == 2) else [13, 31]
            blocks_displacement_r = [0, 104]
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

        grid = self._proc_g
        n_grids = self._n_grids
        n_oct = self._n_oct
        comm_w = self._comm_w
        comm_l = self._comm
        rank_w = self._rank_w
        rank_l = self._rank
        is_background = True
        if (grid):
            is_background = False
        if (n_grids > 1):
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

            if (not is_background):
                self.update_fg_grids(o_ranges,
                                     ids_octree_contained)
            else:
                if (n_grids > 1):
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
                        ids_octree_contained,
                        # Reconstruction order.
                        rec_ord = 2):
        """Method which update the non diagonal blocks relative to the
           foreground grids.

           Arguments:
                o_ranges (tuple) : the initial and final octants managed by the
                                   current process.
                ids_octree_contained (list) : list of the indices of the octants
                                              contained in the current process."""

        octree = self._octree
        comm_l = self._comm
        dimension = self._dim
        grid = self._proc_g
        o_ranges = self.get_ranges()
        # Background bilinear transformation "alpha\" and \"beta\".
        b_alpha = self.get_trans(0)[1]
        b_beta = self.get_trans(0)[2]
        # Current bilinear transformation \"alpha\" and \"beta\".
        c_alpha = self.get_trans(grid)[1]
        c_beta = self.get_trans(grid)[2]
        ex_sols = []

        #start = time.time()
        list_edg = list(self._n_edg)
        # Length list edg.
        l_l_edg = len(list_edg)
        # TODO: for the moment, interaction between grids of the foreground is
        #       implemented only as creation of \"inter-communicators\". Do al-
        #       so the remaining parts.
        list_edg = [list_edg[i] for i in xrange(0, l_l_edg) if
                    int(list_edg[i][0].item(0)) == grid]
        # Length list edg (new).
        # TODO: add a control if there are no element covered by a certain grid of
        #       foreground (if there are too few octants in general, for example
        #       the case I am running with refs 3, 3, 3).
        l_l_edg = len(list_edg)
        #print(l_l_edg)
        #print(list_edg)
        # Length key.
        l_k = list_edg[0][0].size
        # Length stencil.
        l_s = list_edg[0][1].size
        keys = numpy.array([list_edg[i][0] for i in
                            range(0, l_l_edg)]).reshape(l_l_edg, l_k)
        stencils = numpy.array([list_edg[i][1] for i in
                                range(0, l_l_edg)]).reshape(l_l_edg, l_s)
        centers = [(stencils[i][1 : dimension + 1]) for i in range(0, l_l_edg)]
        n_centers = len(centers)
        t_centers_inv = []
        uint32_max = numpy.iinfo(numpy.uint32).max
        local_idxs = numpy.zeros(shape = (n_centers, ), \
                                 dtype = numpy.uint32)
        global_idxs = numpy.zeros(shape = (n_centers, ), \
                                  dtype = numpy.uint32)
        t_center = numpy.zeros(shape = (1, 3), \
                               dtype = numpy.float64)
        t_center_inv = numpy.zeros(shape = (1, 3), \
                                   dtype = numpy.float64)

        # Code hoisting.
        apply_bil_mapping = utilities.apply_bil_mapping
        apply_bil_mapping_inv = utilities.apply_bil_mapping_inv
        find_right_neighbours = self.new_find_right_neighbours
        b_c = utilities.bil_coeffs
        apply_rest_prol_ops = self.new_apply_rest_prol_ops
        mask_octant = self.mask_octant
        narray = numpy.array
        ncopy = numpy.copy
        get_center = octree.get_center
        get_point_owner_idx = octree.get_point_owner_idx
        get_points_local_ring = utilities.get_points_local_ring
        solution = utilities.exact_sol

        # Lambda function.
        f_r_n = lambda x : find_right_neighbours(x[0]       ,
                                                 o_ranges[0],
                                                 x[1]       ,
                                                 True       ,
                                                 x[2])

        for i in xrange(0, n_centers):
            apply_bil_mapping(narray([centers[i]]),
                              b_alpha             ,
                              b_beta              ,
                              t_center            ,
                              dim = 2)
            ex_sol = solution(t_center,
                              b_alpha ,
                              b_beta  ,
                              dim = 3 ,
                              apply_mapping = False)
            ex_sols.append(ex_sol[0])
            apply_bil_mapping_inv(t_center    ,
                                  c_alpha     ,
                                  c_beta      ,
                                  t_center_inv,
                                  dim = 2)
            # \"Numpy\" copy \"t_center_inv\".
            n_c_t_c_i = ncopy(t_center_inv[0])
            t_centers_inv.append(n_c_t_c_i)
            local_idxs[i] = get_point_owner_idx(t_center_inv[0])
            global_idxs[i] = local_idxs[i]
            if (local_idxs[i] != uint32_max):
                global_idxs[i] += o_ranges[0]
        idxs = numpy.where(numpy.logical_and((global_idxs >=
                                              ids_octree_contained[0]),
                                             (global_idxs <=
                                              ids_octree_contained[1])))
        # \"idxs[0]\" because is a numpy array, so to select the array we have
        # to use the index notation.
        for idx in idxs[0]:
            oct_center, \
            n_oct_center  = get_center(local_idxs[idx]   ,
                                       ptr_octant = False,
                                       also_numpy_center = True)
            #if (keys[idx][1] == 639):
            #    print(list_edg[idx])
            #    print(local_idxs[idx])
            h = octree.get_area(local_idxs[idx])
            # TODO: pass also \"h\" of the background octants to be sure of their
            #       dimension carachteristic in case more complicated (here is just
            #       two time the \"h\" of the foreground octants).
            self._h_s_inter_on_board.append(stencils[idx][0])
            self._f_on_borders_exact.append(ex_sols[idx])
            n_rec_sol = solution(narray([n_oct_center]),
                                 c_alpha               ,
                                 c_beta                ,
                                 dim = 2               ,
                                 apply_mapping = True)
            rec_sol = n_rec_sol[0]
            if (rec_ord == 2):
                # TODO: check correctness of the ring and of the indices found
                #       (understand if \"n_cs_n_is[1]\" has to be masked or not).
                oct_ring = get_points_local_ring(t_centers_inv[idx],
                                                 n_oct_center)
                # Neighbour centers neighbour indices.
                n_cs_n_is = f_r_n((local_idxs[idx],
                                   oct_ring       ,
                                   t_centers_inv[idx]))
                #if (keys[idx][1] == 148):
                #    print(oct_ring)
                #    print(local_idxs[idx])
                #    print(n_cs_n_is)
                #if (keys[idx][1] == 148):
                #    print(n_cs_n_is)
                rec_sols = solution(n_cs_n_is[0],
                                    c_alpha     ,
                                    c_beta      ,
                                    dim = 2     ,
                                    apply_mapping = True)
                coeffs = b_c(n_cs_n_is[0],
                             t_centers_inv[idx])
                #if (keys[idx][1] == 148):
                #    print(coeffs)
                rec_sol_0 = rec_sols[0] * coeffs[0]
                rec_sol_1 = rec_sols[1] * coeffs[1]
                rec_sol_2 = rec_sols[2] * coeffs[2]
                rec_sol_3 = 0.0
                if (n_cs_n_is[0].shape[0] == 4):
                    rec_sol_3 = rec_sols[3] * coeffs[3]
                rec_sol = rec_sol_0 + rec_sol_1 + rec_sol_2 + rec_sol_3
            self._f_on_borders.append(rec_sol)

            # Checkout how the \"stencil\" is created in the function
            # \"create_mask\".
            displ = 1 + dimension
            step = 2
            row_indices = []
            col_indices = []
            col_values = []
            for i in xrange(displ, len(stencils[idx]), step):
                row_index = int(stencils[idx][i])
                if (row_index == -1):
                    break
                # Masked row index (in function \"create_mask\", they are not
                # still masked).
                m_row_index = self._ngn[row_index]
                row_indices.append(m_row_index)
                value_to_multiply = stencils[idx][i + 1]
                # Current coefficients.
                c_coeffs = value_to_multiply
                #nsolution = utilities.exact_sol(narray([[stencils[idx][0],
                #                                         stencils[idx][1]]]),
                #                                       b_alpha              ,
                #                                       b_beta               ,
                #                                       dim = 2              ,
                #                                       apply_mapping = True)
                #c_coeffs = c_coeffs * nsolution * -1.0
                if (rec_ord == 2):
                    n_copy_coeffs = numpy.copy(coeffs * value_to_multiply)
                    c_coeffs = n_copy_coeffs.tolist()
                    #if (keys[idx][1] == 148):
                    #    print(stencils[idx])
                    #    print(value_to_multiply)
                    #    print(c_coeffs)
                    #nsolutions = solution(n_cs_n_is[0],
                    #                      c_alpha     ,
                    #                      c_beta      ,
                    #                      dim = 2     ,
                    #                      apply_mapping = True)
                    #n_sol_0 = nsolutions[0] * coeffs[0] * -1
                    #n_sol_1 = nsolutions[1] * coeffs[1] * -1
                    #n_sol_2 = nsolutions[2] * coeffs[2] * -1
                    #n_sol_3 = 0.0
                    #if (n_cs_n_is[0].shape[0] == 4):
                    #    n_sol_3 = nsolutions[3] * coeffs[3] * -1
                    #n_sol = n_sol_0 + n_sol_1 + n_sol_2 + n_sol_3
                    #c_coeffs = n_sol
                col_values.append(c_coeffs)
            if (rec_ord == 2):
                col_indices.extend(n_cs_n_is[1])
            else:
                col_index = mask_octant(global_idxs[idx])
                col_indices.append(col_index)
            if (row_indices):
                #insert_mode = PETSc.InsertMode.ADD_VALUES
                #self._rhs.setValues(row_indices,
                #                    col_values ,
                #                    insert_mode)
                apply_rest_prol_ops(row_indices,
                                    col_indices,
                                    col_values)

        msg = "Updated prolongation blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def check_bg_bad_diamond_point(self                ,
                                   n_point             , # \"numpy\" point
                                   idx_ptr_owner       ,
                                   foreground          ,
                                   h_inter             ,
                                   codim               ,
                                   iface               ,
                                   dimension = 2       ,
                                   is_ptr_owner = False,
                                   # \"numpy\" background octant center.
                                   n_bg_center = numpy.zeros(shape = (1, 3), \
                                                        dtype = numpy.float64)):
        octree = self._octree
        get_area = octree.get_area
        b_alpha = self.get_trans(0)[1]
        b_beta = self.get_trans(0)[2]
        bad_diamond_point = False
        n_t_a_01 = numpy.zeros(shape = (1, dimension), \
                               dtype = numpy.float64)
        n_t_a_02 = numpy.zeros(shape = (1, dimension), \
                               dtype = numpy.float64)
        # Edge size of the octant of the background, owner of \"n_point\".
        h_bg_inter = get_area(idx_ptr_owner,
                              is_ptr = is_ptr_owner)

        apply_bil_mapping = utilities.apply_bil_mapping
        get_center = octree.get_center
        get_area = octree.get_area
        is_point_inside_polygon = utilities.is_point_inside_polygon
        neigh_inter_center = utilities.neigh_inter_center
        ncopyto = numpy.copyto

        n_i_c = lambda x : neigh_inter_center(x         ,
                                              h_bg_inter,
                                              codim     ,
                                              iface)

        n_oct_center = n_bg_center

        if (not is_ptr_owner):
            oct_center, \
            n_oct_center  = get_center(idx_ptr_owner     ,
                                       ptr_octant = False,
                                       also_numpy_center = True)


        ncopyto(n_t_a_01[0][: dimension], \
                n_oct_center[: dimension])

        apply_bil_mapping(n_t_a_01,
                          b_alpha ,
                          b_beta  ,
                          n_t_a_02,
                          dimension)

        is_in_fg = is_point_inside_polygon(n_t_a_02,
                                           foreground)

        if (is_in_fg):
            bad_diamond_point = True
            ncopyto(n_point[0][: dimension],
                    n_i_c(n_point)[0][: dimension])

        return bad_diamond_point
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

        octree = self._octree
        comm_l = self._comm
        dimension = self._dim
        proc_grid = self._proc_g

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
        stencils = numpy.array([list_edg[i][1] for i in
                            xrange(0, l_l_edg)]).reshape(l_l_edg, l_s)
        # Background bilinear transformation "alpha\" and \"beta\".
        b_alpha = self.get_trans(0)[1]
        b_beta = self.get_trans(0)[2]

        # Code hoisting.
        apply_bil_mapping = utilities.apply_bil_mapping
        apply_bil_mapping_inv = utilities.apply_bil_mapping_inv
        find_right_neighbours = self.new_find_right_neighbours
        find_neighbours = octree.find_neighbours
        least_squares = utilities.bil_coeffs
        apply_rest_prol_ops = self.new_apply_rest_prol_ops
        narray = numpy.array
        ncopy = numpy.copy
        ncopyto = numpy.copyto
        get_point_owner_idx = octree.get_point_owner_idx
        neigh_inter_center = utilities.neigh_inter_center
        get_interface_coefficients_1_order = self.get_interface_coefficients_1_order
        get_center = octree.get_center
        get_ghost_octant = octree.get_ghost_octant
        get_ghost_global_idx = octree.get_ghost_global_idx
        bil_coeffs = utilities.bil_coeffs
        check_bg_bad_diamond_point = self.check_bg_bad_diamond_point
        mask_octant = self.mask_octant
        solution = utilities.exact_sol
        get_codim_iface = utilities.get_codim_iface
        uint32_max = numpy.iinfo(numpy.uint32).max

        b_c = lambda x : bil_coeffs(x[0],
                                    x[1])
        # \"numpy\" temporary array 01.
        n_t_a_01 = numpy.zeros(shape = (1, 3), \
                               dtype = numpy.float64)
        # \"numpy\" temporary array 02.
        n_t_a_02 = numpy.zeros(shape = (1, 3), \
                               dtype = numpy.float64)
        # \"numpy\" temporary array 03.
        n_t_a_03 = numpy.zeros(shape = (1, 3), \
                               dtype = numpy.float64)

        for i in xrange(0, l_l_edg):
            grid = keys[i][0]
            # Foreground intersection size.
            h_inter = stencils[i][0]
            # Transformed foreground.
            t_foreground = self._t_foregrounds[grid - 1]
            # \"numpy\" transformed foreground.
            n_t_foreground = narray([t_foreground])
            # Current bilinear transformation \"alpha\" and \"beta\".
            c_alpha = self.get_trans(grid)[1]
            c_beta = self.get_trans(grid)[2]
            # One node of the intersection on foreground boundaries.
            if (keys[i][3] == 1):
                #pass
                t_centers_inv = []
                l_t_indices_inv = []
                t_nodes_inv = []
                # Getting coordinates of the intersection node on the foreground
                # border.
                displ = 2
                ncopyto(n_t_a_03[0][: dimension], \
                        stencils[i][displ : displ + dimension])
                #----check bil mapping inv-----
                #print("alpha " +str(c_alpha))
                #print("beta " + str(c_beta))
                apply_bil_mapping(n_t_a_03,
                                  c_alpha ,
                                  c_beta  ,
                                  n_t_a_01,
                                  dimension)
                apply_bil_mapping_inv(n_t_a_01,
                                      c_alpha ,
                                      c_beta  ,
                                      n_t_a_02,
                                      dimension)
                if (abs(n_t_a_02[0][0] - n_t_a_03[0][0] > 1.0e-15) and
                    abs(n_t_a_02[0][1] - n_t_a_03[0][1]) > 1.0e-15):
                    print("before " + str(n_t_a_03))
                    print("middle " + str(n_t_a_01))
                    print("after " + str(n_t_a_02))
                #----check bil mapping inv-----
                apply_bil_mapping(n_t_a_03,
                                  c_alpha ,
                                  c_beta  ,
                                  n_t_a_01,
                                  dimension)
                apply_bil_mapping_inv(n_t_a_01,
                                      b_alpha ,
                                      b_beta  ,
                                      n_t_a_02,
                                      dimension)
                local_idx = get_point_owner_idx(n_t_a_02[0])
                global_idx = local_idx
                if (local_idx != uint32_max):
                    global_idx += o_ranges[0]
                    m_index = mask_octant(global_idx)
                    cell_center, \
                    n_cell_center = get_center(local_idx,
                                               False    ,
                                               True)
                    n_n_cell_center = narray([n_cell_center])
                    #apply_bil_mapping(n_n_cell_center,
                    #                  b_alpha        ,
                    #                  b_beta         ,
                    #                  n_t_a_01       ,
                    #                  dimension)
                    #apply_bil_mapping_inv(n_t_a_01,
                    #                      c_alpha ,
                    #                      c_beta  ,
                    #                      n_t_a_02,
                    #                      dimension)
                    #c_n_oct_center = ncopy(n_t_a_02[0])
                    c_n_oct_center = ncopy(n_cell_center)
                    t_centers_inv.append(c_n_oct_center[: dimension])
                    l_t_indices_inv.append(m_index)
                    neighs, ghosts = ([] for i in range(0, 2))
                    for codim in xrange(1, 3):
                        for iface in xrange(0, 4):
                            (neighs, \
                             ghosts) = find_neighbours(local_idx,
                                                       iface    ,
                                                       codim    ,
                                                       neighs   ,
                                                       ghosts)
                            n_neighs = len(neighs)
                            if (neighs):
                                # Distance center node.
                                d_c_n = 0.0
                                for j in xrange(0, n_neighs):
                                    # Neighbour is into the same process, so is local.
                                    if (not ghosts[j]):
                                        by_octant = False
                                        index = neighs[j]
                                        m_index = mask_octant(index + o_ranges[0])
                                        py_ghost_oct = index
                                    else:
                                        by_octant = True
                                        # In this case, the quas(/oc)tree is no more local into
                                        # the current process, so we have to find it globally.
                                        index = get_ghost_global_idx(neighs[j])
                                        # \".index\" give us the \"self._global_ghosts\" index
                                        # that contains the index of the global ghost quad(/oc)-
                                        # tree previously found and stored in \"index\".
                                        py_ghost_oct = get_ghost_octant(neighs[j])
                                        m_index = mask_octant(index)
                                    if (m_index != -1):
                                        cell_center, \
                                        n_cell_center = get_center(py_ghost_oct,
                                                                   by_octant   ,
                                                                   True)
                                        n_n_cell_center = narray([n_cell_center])
                                        #apply_bil_mapping(n_n_cell_center,
                                        #                  b_alpha        ,
                                        #                  b_beta         ,
                                        #                  n_t_a_01       ,
                                        #                  dimension)
                                        #apply_bil_mapping_inv(n_t_a_01,
                                        #                      c_alpha ,
                                        #                      c_beta  ,
                                        #                      n_t_a_02,
                                        #                      dimension)
                                        # Temporary distance.
                                        #t_d = numpy.linalg.norm(n_t_a_02[0][: dimension] - \
                                        #                        n_t_a_03[0][: dimension])
                                        t_d = numpy.linalg.norm(n_cell_center[: dimension] - \
                                                                n_t_a_02[0][: dimension])
                                        # \"j\" == 0...first neighbour.
                                        if (not j):
                                            d_c_n = t_d
                                            #c_n_oct_center = ncopy(n_t_a_02[0])
                                            c_n_oct_center = ncopy(n_cell_center)
                                            t_centers_inv.append(c_n_oct_center[: dimension])
                                            l_t_indices_inv.append(m_index)
                                        # Second neighbour case.
                                        else:
                                            if (t_d < d_c_n):
                                                d_c_n = t_d
                                                c_n_oct_center = ncopy(n_cell_center)
                                                #c_n_oct_center = ncopy(n_t_a_02[0])
                                                t_centers_inv[-1][: dimension] = \
                                                c_n_oct_center[: dimension]
                                                l_t_indices_inv[-1] = m_index
                    coeffs = utilities.least_squares(narray(t_centers_inv),
                                                     n_t_a_02[0][: dimension])
                                                     #n_t_a_03[0][: dimension])
                    # Coordinates of the node on the foreground boundary.
                    cs_n = stencils[i][displ : displ + dimension]
                    ## \"Numpy\" coordinates of the node on the foreground boun-
                    ## dary.
                    n_cs_n = narray([cs_n])
                    ex_sol = solution(n_cs_n,
                                      c_alpha              ,
                                      c_beta               ,
                                      dim = dimension      ,
                                      apply_mapping = True)
                    self._f_on_borders_exact.append(ex_sol[0])
                    self._h_s_inter_on_board.append(h_inter)
                    rec_sols = solution(narray(t_centers_inv),
                                        #c_alpha              ,
                                        #c_beta               ,
                                        b_alpha              ,
                                        b_beta               ,
                                        dim = dimension      ,
                                        apply_mapping = True)
                    rec_sol = 0
                    for k in xrange(0, rec_sols.shape[0]):
                        rec_sol += rec_sols[k] * coeffs[k]
                    self._f_on_borders.append(rec_sol)
                    #rec_sol = rec_sol * stencils[i][1]
                    #insert_mode = PETSc.InsertMode.ADD_VALUES
                    #self._rhs.setValues([keys[i][1], keys[i][2]] ,
                    #                    [rec_sol, rec_sol * -1.0],
                    #                    insert_mode)
                    # Outer normal coeffs.
                    o_n_coeffs = coeffs * stencils[i][1]
                    # Inner normal coeffs.
                    i_n_coeffs = o_n_coeffs * -1.0
                    col_values = []
                    col_values.append(i_n_coeffs.tolist())
                    col_values.append(o_n_coeffs.tolist())
                    row_indices = []
                    row_indices.append(keys[i][1])
                    row_indices.append(keys[i][2])
                    col_indices = l_t_indices_inv
                    #print(col_indices)
                    apply_rest_prol_ops(row_indices,
                                        col_indices,
                                        col_values)
                    #insert_mode = PETSc.InsertMode.ADD_VALUES
                    #self._rhs.setValues(keys[i][1],
                    #                    ex_sol[0] * 1.0 * stencils[i][1],
                    #                    insert_mode)
                    #self._rhs.setValues(keys[i][2],
                    #                    ex_sol[0] * -1.0 * stencils[i][1],
                    #                    insert_mode)
            # Two nodes on foreground boundaries.
            elif (keys[i][3] == 2):
                t_centers_inv = []
                l_t_indices_inv = []
                t_nodes_inv = []
                # Getting coordinates of the first neighbour (the one of the
                # intersection) of the rings of the nodes (it will be the same
                # for both the nodes).
                displ = 1 + (dimension * 2)
                ncopyto(n_t_a_03[0][: dimension], \
                        stencils[i][displ : displ + dimension])
                apply_bil_mapping(n_t_a_03,
                                  c_alpha ,
                                  c_beta  ,
                                  n_t_a_01,
                                  dimension)
                apply_bil_mapping_inv(n_t_a_01,
                                      b_alpha ,
                                      b_beta  ,
                                      n_t_a_02,
                                      dimension)
                local_idx = get_point_owner_idx(n_t_a_02[0])
                global_idx = local_idx
                if (local_idx != uint32_max):
                    global_idx += o_ranges[0]
                    m_index = mask_octant(global_idx)
                    cell_center, \
                    n_cell_center = get_center(local_idx,
                                               False    ,
                                               True)
                    n_n_cell_center = narray([n_cell_center])
                    #apply_bil_mapping(n_n_cell_center,
                    #                  b_alpha        ,
                    #                  b_beta         ,
                    #                  n_t_a_01       ,
                    #                  dimension)
                    #apply_bil_mapping_inv(n_t_a_01,
                    #                      c_alpha ,
                    #                      c_beta  ,
                    #                      n_t_a_02,
                    #                      dimension)
                    #c_n_oct_center = ncopy(n_t_a_02[0])
                    c_n_oct_center = ncopy(n_cell_center)
                    t_centers_inv.append(c_n_oct_center[: dimension])
                    l_t_indices_inv.append(m_index)
                    neighs, ghosts = ([] for i in range(0, 2))
                    for codim in xrange(1, 3):
                        for iface in xrange(0, 4):
                            (neighs, \
                             ghosts) = find_neighbours(local_idx,
                                                       iface    ,
                                                       codim    ,
                                                       neighs   ,
                                                       ghosts)
                            n_neighs = len(neighs)
                            if (neighs):
                                # Distance center node.
                                d_c_n = 0.0
                                for j in xrange(0, n_neighs):
                                    # Neighbour is into the same process, so is local.
                                    if (not ghosts[j]):
                                        by_octant = False
                                        index = neighs[j]
                                        m_index = mask_octant(index + o_ranges[0])
                                        py_ghost_oct = index
                                    else:
                                        by_octant = True
                                        # In this case, the quas(/oc)tree is no more local into
                                        # the current process, so we have to find it globally.
                                        index = get_ghost_global_idx(neighs[j])
                                        # \".index\" give us the \"self._global_ghosts\" index
                                        # that contains the index of the global ghost quad(/oc)-
                                        # tree previously found and stored in \"index\".
                                        py_ghost_oct = get_ghost_octant(neighs[j])
                                        m_index = mask_octant(index)
                                    if (m_index != -1):
                                        cell_center, \
                                        n_cell_center = get_center(py_ghost_oct,
                                                                   by_octant   ,
                                                                   True)
                                        n_n_cell_center = narray([n_cell_center])
                                        #apply_bil_mapping(n_n_cell_center,
                                        #                  b_alpha        ,
                                        #                  b_beta         ,
                                        #                  n_t_a_01       ,
                                        #                  dimension)
                                        #apply_bil_mapping_inv(n_t_a_01,
                                        #                      c_alpha ,
                                        #                      c_beta  ,
                                        #                      n_t_a_02,
                                        #                      dimension)
                                        # Temporary distance.
                                        #t_d = numpy.linalg.norm(n_t_a_02[0][: dimension] - \
                                        #                        n_t_a_03[0][: dimension])
                                        t_d = numpy.linalg.norm(n_t_a_02[0][: dimension] - \
                                                                n_cell_center[: dimension])
                                        # \"j\" == 0...first neighbour.
                                        if (not j):
                                            d_c_n = t_d
                                            #c_n_oct_center = ncopy(n_t_a_02[0])
                                            c_n_oct_center = ncopy(n_cell_center)
                                            t_centers_inv.append(c_n_oct_center[: dimension])
                                            l_t_indices_inv.append(m_index)
                                        # Second neighbour case.
                                        else:
                                            if (t_d < d_c_n):
                                                d_c_n = t_d
                                                c_n_oct_center = ncopy(n_cell_center)
                                                #c_n_oct_center = ncopy(n_t_a_02[0])
                                                t_centers_inv[-1][: dimension] = \
                                                c_n_oct_center[: dimension]
                                                l_t_indices_inv[-1] = m_index
                    l_s_coeffs = utilities.least_squares(narray(t_centers_inv),
                                                         n_t_a_02[0][: dimension])
                                                         #n_t_a_03[0][: dimension])
                    ncopyto(n_t_a_03[0][: dimension], \
                            stencils[i][1 : 3])
                    apply_bil_mapping(n_t_a_03,
                                      c_alpha ,
                                      c_beta  ,
                                      n_t_a_01,
                                      dimension)
                    apply_bil_mapping_inv(n_t_a_01,
                                          b_alpha ,
                                          b_beta  ,
                                          n_t_a_02,
                                          dimension)
                    l_s_coeffs_node_0 = utilities.least_squares(narray(t_centers_inv),
                                                                n_t_a_02[0][: dimension])
                    ncopyto(n_t_a_03[0][: dimension], \
                            stencils[i][3 : 5])
                    apply_bil_mapping(n_t_a_03,
                                      c_alpha ,
                                      c_beta  ,
                                      n_t_a_01,
                                      dimension)
                    apply_bil_mapping_inv(n_t_a_01,
                                          b_alpha ,
                                          b_beta  ,
                                          n_t_a_02,
                                          dimension)
                    l_s_coeffs_node_1 = utilities.least_squares(narray(t_centers_inv),
                                                                n_t_a_02[0][: dimension])
                    displ = 1 + (2 * dimension)
                    # Coordinates of the node on the foreground boundary.
                    cs_n = stencils[i][displ : displ + dimension]
                    # \"Numpy\" coordinates of the node on the foreground boun-
                    # dary.
                    n_cs_n = narray([cs_n])
                    ex_sol = solution(n_cs_n,
                                      c_alpha              ,
                                      c_beta               ,
                                      dim = dimension      ,
                                      apply_mapping = True)
                    self._f_on_borders_exact.append(ex_sol[0])
                    self._h_s_inter_on_board.append(h_inter)
                    rec_sols = solution(narray(t_centers_inv),
                                        #c_alpha              ,
                                        #c_beta               ,
                                        b_alpha              ,
                                        b_beta               ,
                                        dim = dimension      ,
                                        apply_mapping = True)
                    rec_sol = 0
                    for k in xrange(0, rec_sols.shape[0]):
                        rec_sol += rec_sols[k] * l_s_coeffs[k]
                    self._f_on_borders.append(rec_sol)

                    nodes_inter = [stencils[i][1 : 3], stencils[i][3 : 5]]
                    owners_centers = [stencils[i][5 : 7], stencils[i][11 : 13]]
                    bil_coeffs_empty = numpy.array([[], []])
                    n_coeffs     , \
                    coeffs_node_1, \
                    coeffs_node_0 = self.get_interface_coefficients(0                        ,
                                                                    dimension                ,
                                                                    nodes_inter              ,
                                                                    owners_centers           ,
                                                                    bil_coeffs_empty         ,
                                                                    use_inter = False        ,
                                                                    h_given = h_inter        ,
                                                                    n_axis_given = keys[i][5],
                                                                    n_value_given = keys[i][6],
                                                                    grid = keys[i][0])
                    #insert_mode = PETSc.InsertMode.ADD_VALUES
                    #self._rhs.setValues(keys[i][1],
                    #                    ex_sol[0] * -1.0 * n_coeffs[0],
                    #                    insert_mode)
                    #n_cs_n = narray([nodes_inter[0]])
                    #ex_sol = solution(n_cs_n,
                    #                  c_alpha              ,
                    #                  c_beta               ,
                    #                  dim = dimension      ,
                    #                  apply_mapping = True)
                    #self._rhs.setValues(keys[i][1],
                    #                    ex_sol[0] * -1.0 * coeffs_node_0,
                    #                    insert_mode)
                    #n_cs_n = narray([nodes_inter[1]])
                    #ex_sol = solution(n_cs_n,
                    #                  c_alpha              ,
                    #                  c_beta               ,
                    #                  dim = dimension      ,
                    #                  apply_mapping = True)
                    #insert_mode = PETSc.InsertMode.ADD_VALUES
                    #self._rhs.setValues(keys[i][1],
                    #                    ex_sol[0] * -1.0 * coeffs_node_1,
                    #                    insert_mode)
                    #coeff_o = rec_sol * -1.0 * n_coeffs[0]
                    #insert_mode = PETSc.InsertMode.ADD_VALUES
                    #self._rhs.setValues([keys[i][1]],
                    #                    [coeff_o]   ,
                    #                    insert_mode)
                    coeffs_ghost = l_s_coeffs * n_coeffs[0]
                    coeffs_ghost_node_0 = l_s_coeffs_node_0 * n_coeffs[3]
                    coeffs_ghost_node_1 = l_s_coeffs_node_1 * n_coeffs[2]
                    columns = []
                    values = []
                    values.append(n_coeffs[1])
                    values.extend(coeffs_ghost.tolist())
                    values.extend(coeffs_ghost_node_0.tolist())
                    values.extend(coeffs_ghost_node_1.tolist())
                    columns.append(keys[i][1])
                    columns.extend(l_t_indices_inv)
                    columns.extend(l_t_indices_inv)
                    columns.extend(l_t_indices_inv)
                    apply_rest_prol_ops([keys[i][1]],
                                        columns     ,
                                        values)

        msg = "Updated restriction blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def new_find_right_neighbours(self                  ,
                                  current_octant        ,
                                  start_octant          ,
                                  inter_ring            ,
                                  with_node = False     ,
                                  n_node = None         ,
                                  is_node_on_f_b = False,
                                  # also covered indices
                                  a_cov_indices = False):
        if (current_octant == "b_boundary"):
            # A \"numpy\" empty array (size == 0) of shape (2, 0).
            n_e_array = numpy.array([[], []])
            return (n_e_array, n_e_array)
        octree = self._octree
        py_oct = octree.get_octant(current_octant)
        centers = []
        indices = []
        cov_indices = []
        grid = self._proc_g
        l_ring = 3
        dimension = self._dim
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        t_background = numpy.array([self._t_background])
        # Ghosts' deplacement.
        g_d = 0
        for i in xrange(0, grid):
            g_d = g_d + self._oct_f_g[i]
        # Current center.
        c_c = octree.get_center(current_octant)[: dimension]
        h = octree.get_area(current_octant)

        index = current_octant
        # Current mask index.
        c_m_index = self.mask_octant(index + start_octant)
        #if (not grid):
        #    if ((index + g_d) != (index + start_octant)):
        #        print("g_d = " + str(index + g_d))
        #        print("start = " + str(index + start_octant))

        neighs, ghosts = ([] for i in range(0, 2))

        #Code hoisting.
        find_neighbours = octree.find_neighbours
        mask_octant = self.mask_octant
        get_center = octree.get_center
        get_ghost_global_idx = octree.get_ghost_global_idx
        get_ghost_octant = octree.get_ghost_octant
        neighbour_centers = self.neighbour_centers
        is_point_inside_polygon = utilities.is_point_inside_polygon
        apply_bil_mapping = utilities.apply_bil_mapping
        # Lambda function.
        f_n = lambda x, y : find_neighbours(current_octant,
                                            x             ,
                                            y             ,
                                            neighs        ,
                                            ghosts)

        for i in xrange(0, l_ring):
            # Codimension = 1, looping just on the faces, but the second element
            # of the local ring will be the neighbour of node.
            codim = 2 if (i == 1) else 1
            # Index of current face or node.
            face_node = inter_ring[i]

            (neighs, ghosts) = f_n(face_node, codim)
            n_neighs = len(neighs)
            # Check if it is really a neighbour of edge or node. If not,
            # it means that we are near the boundary.
            if (neighs):
                # Distance center node.
                d_c_n = 0.0
                for j in xrange(0, n_neighs):
                    # Neighbour is into the same process, so is local.
                    if (not ghosts[j]):
                        by_octant = False
                        index = neighs[j]
                        m_index = mask_octant(index + start_octant)
                        cov_index = index + start_octant
                        py_ghost_oct = index
                    else:
                        by_octant = True
                        # In this case, the quas(/oc)tree is no more local into
                        # the current process, so we have to find it globally.
                        index = get_ghost_global_idx(neighs[j])
                        # \".index\" give us the \"self._global_ghosts\" index
                        # that contains the index of the global ghost quad(/oc)-
                        # tree previously found and stored in \"index\".
                        py_ghost_oct = get_ghost_octant(neighs[j])
                        m_index = mask_octant(index + g_d)
                        cov_index = index + g_d
                    # TODO: problem here: \"m_octant\" = -1 will be added to the
                    #       matrix, but PETSc will not add any coefficients, be-
                    #       cause \"-1\" is an its special sign to say to not insert
                    #       in that column/row.
                    #if (not grid):
                    #    print(m_index)
                    cell_center, \
                    n_cell_center = get_center(py_ghost_oct,
                                               by_octant   ,
                                               True)
                    if (with_node):
                        # Temporary distance.
                        t_d = numpy.linalg.norm(n_cell_center - \
                                                n_node)
                        # \"j\" == 0...first neighbour.
                        if (not j):
                            d_c_n = t_d
                            centers.append(cell_center[: dimension])
                            indices.append(m_index)
                            cov_indices.append(cov_index)
                        # Second neighbour case.
                        else:
                            if (t_d < d_c_n):
                                d_c_n = t_d
                                centers[-1] = cell_center[: dimension]
                                indices[-1] = m_index
                                cov_indices[-1] = cov_index
                    else:
                        centers.append(cell_center[: dimension])
                        indices.append(m_index)
                        cov_indices.append(cov_index)
            # ...we need to evaluate boundary values (background) or not to
            # consider the indices and centers found (foreground).
            else:
                if (grid and is_node_on_f_b):
                    # TODO: rewrite function \"neighbour_centers\", because it
                    #       is not well done.
                    border_center, \
                    numpy_border_center = neighbour_centers(c_c      ,
                                                            codim    ,
                                                            face_node,
                                                            h        ,
                                                            r_a_n_d = True)
                    indices.append(-1)
                    cov_indices.append(-1)
                    centers.append(border_center[: dimension])
                    # TODO: for the moment, we do not consider this case. Think
                    #       better about it.
                    #t_center = numpy.zeros(shape = (1, 3), dtype = numpy.float64)
                    #apply_bil_mapping(numpy.array([numpy_border_center]),
                    #                  alpha                             ,
                    #                  beta                              ,
                    #                  t_center                          ,
                    #                  dim = 2)
                    #check = is_point_inside_polygon(t_center    ,
                    #                                t_background)

                    #if (check):
                    #    indices.append(-1)
                    #else:
                    #    indices.append("outside_bg")
                    #centers.append(border_center[: dimension])

        centers.append(c_c)

        indices.append(c_m_index)
        cov_indices.append(current_octant + start_octant)

        numpy_centers = numpy.array(centers)
        #if (grid):
        #    if (current_octant== 188):
        #        print(cov_indices)

        if (a_cov_indices):
            return (numpy_centers, indices, cov_indices)

        return (numpy_centers, indices)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def fill_mat(self             ,
                 inter            ,
                 owners_centers   ,
                 n_cs_n_is        ,
                 r_indices        ,
                 labels           ,
                 g_o_norms_inter  ,
                 m_g_o_norms_inter,
                 coeffs_nodes     ,
                 n_polygon        ,
                 n_coeffs         ,
                 nodes_inter      ,
                 n_nodes_on_f_b   ,
                 node_on_f_b      ,
                 rings):

        grid = self._proc_g
        octree = self._octree
        mask_octant = self.mask_octant
        dimension = self._dim
        coeffs_node_0 = coeffs_nodes[0]
        coeffs_node_1 = coeffs_nodes[1]
        is_background = False if (grid) else True

        insert_mode = PETSc.InsertMode.ADD_VALUES
        narray = numpy.array
        b_alpha = self.get_trans(0)[1]
        b_beta = self.get_trans(0)[2]
        # Setting exact solution for bilinear background centers wich are penalized
        # by the foreground and used to interpolate nodes of background intersections.
        value_to_store = 0
        if (not grid):
            #print(n_cs_n_is)
            for i in xrange(0, 2):
                for j in xrange(0, len(n_cs_n_is[i][1])):
                    if (n_cs_n_is[i][1][j] == -1):
                        ##print("bongo")
                        #values_rhs = []
                        #indices_rhs = []
                        #nsolution = utilities.exact_sol(narray([[n_cs_n_is[i][0][j][0],
                        #                                         n_cs_n_is[i][0][j][1]]]),
                        #                                b_alpha              ,
                        #                                b_beta               ,
                        #                                dim = 2              ,
                        #                                apply_mapping = True)
                        #mult = 1.0
                        #if (labels[0]):
                        #    mult = -1.0
                        #e_sol = nsolution[0]
                        #e_sol_coeff = coeffs_nodes[i][j]
                        #e_sol = mult * e_sol * e_sol_coeff
                        #values_rhs.append(e_sol)
                        #indices_rhs.append(r_indices[0])
                        #if (len(r_indices) == 2):
                        #    values_rhs.append(e_sol * -1.0)
                        #    indices_rhs.append(r_indices[1])
                        #self._rhs.setValues(indices_rhs,
                        #                    values_rhs,
                        #                    insert_mode)
                        #coeffs_nodes[i][j] = 0.0
                        #mult = -1.0
                        #if (labels[0]):
                        #    mult = 1.0
                        #value_to_store = value_to_store + (coeffs_nodes[i][j] * mult)
                        #coeffs_nodes[i][j] = 0.0
                        p_g_index = n_cs_n_is[i][2][j]
                        #if (p_g_index == 639):
                        #    print(coeffs_nodes[i][j])
                        #    print(n_cs_n_is)
                        #print(n_cs_n_is)
                        #print(p_g_index)
                        #print(n_cs_n_is[i][2])
                        n_polygon = self._g_p_o_f_g[p_g_index]
                        #print(n_cs_n_is)
                        #print("p_g_index " + str(p_g_index) + " n_polygon " + str(n_polygon))
                        #print(self._g_p_o_f_g)
                        key = (n_polygon + 1, \
                               p_g_index    , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0            , \
                               0)

                        stencil = self._edl.get(key)
                        # Sometimes \"stencil\" is equal to \"None\" because
                        # there are values of \"p_g_index\" which correspond to
                        # ghost octant not included in the local octree, and in
                        # the local \"self._edl\".
                        # TODO: I think that with the add of the \"cur_proc_owner\" in
                        #       the function \"fill_mat_and_rhs\", this \"if\" is no
                        #       more useful.
                        if (stencil):
                            #if (p_g_index == 148):
                            #    #print(str(stencil) + " " + str(key))
                            #    print(p_g_index)
                            for q in xrange(0, len(r_indices)):
                                n_p_g_index = g_o_norms_inter[labels[q]]
                                #print(n_p_g_index)
                                displ = 1 + dimension
                                step = 2
                                l_stencil = 21 if (dimension == 2) else 31

                                mult = -1.0
                                if (labels[q]):
                                 mult = 1.0

                                value_to_store = mult * coeffs_nodes[i][j]

                                for k in xrange(displ, l_stencil, step):
                                    if (stencil[k] == n_p_g_index):
                                        #if (p_g_index == 639):
                                        #    if (n_p_g_index == 725):
                                        #        print("prima = " + str(stencil))
                                        #        print("value = " + str(value_to_store))
                                        #print(stencil)
                                        #print("bella")
                                        #print(stencil[k + 1])
                                        stencil[k + 1] = stencil[k + 1] + value_to_store
                                        #if (p_g_index == 639):
                                        #    if (n_p_g_index == 725):
                                        #        print("dopo = " + str(stencil))
                                        break
                        else:
                            key = (n_polygon + 1, \
                                   p_g_index    , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   0            , \
                                   1)
                            stencil = self._edl.get(key)
                            if (stencil):
                                for q in xrange(0, len(r_indices)):
                                    n_p_g_index = g_o_norms_inter[labels[q]]
                                    #print(n_p_g_index)
                                    displ = 1 + dimension
                                    step = 2
                                    l_stencil = 21 if (dimension == 2) else 31

                                    mult = -1.0
                                    if (labels[q]):
                                     mult = 1.0

                                    value_to_store = mult * coeffs_nodes[i][j]
                                    #if (p_g_index == 639):
                                    #    if (n_p_g_index == 725):
                                    #        print(value_to_store)

                                    for k in xrange(displ, l_stencil, step):
                                        if (stencil[k] == n_p_g_index):
                                            #if (p_g_index == 639):
                                            #    if (n_p_g_index == 725):
                                            #        print("prima = " + str(stencil))
                                            #        print("value = " + str(value_to_store))
                                            #print(stencil)
                                            #print("bella")
                                            #print(stencil[k + 1])
                                            stencil[k + 1] = stencil[k + 1] + value_to_store
                                            #if (p_g_index == 639):
                                            #    if (n_p_g_index == 725):
                                            #        print("dopo = " + str(stencil))
                                            break
                                        elif (stencil[k] == -1):
                                            stencil[k] = n_p_g_index
                                            stencil[k + 1] = stencil[k + 1] + value_to_store
                                            break
                                        else:
                                            pass
                                        #if (p_g_index == 639):
                                        #    if (n_p_g_index == 725):
                                        #        print(stencil[k+1])
                                        #        print(stencil)
                            else:
                                h = octree.get_area(inter        ,
                                                    is_ptr = True,
                                                    is_inter = True)
                                l_stencil = 21 if (dimension == 2) else 31
                                stencil = [0, -1] * (l_stencil/2)
                                stencil.append(0)
                                stencil[0] = h
                                #print(n_cs_n_is)
                                for cen_coord in xrange(dimension):
                                    stencil[cen_coord + 1] = n_cs_n_is[i][0][j][cen_coord]
                                displ = 1 + dimension
                                step = 0
                                for q in xrange(0, len(r_indices)):
                                    n_p_g_index = g_o_norms_inter[labels[q]]

                                    mult = -1.0
                                    if (labels[q]):
                                     mult = 1.0

                                    value_to_store = mult * coeffs_nodes[i][j]
                                    #if (p_g_index == 639):
                                    #    if (n_p_g_index == 725):
                                    #        print(value_to_store)
                                    stencil[displ + step] = n_p_g_index
                                    stencil[displ + step + 1] = stencil[displ+step+1] + value_to_store
                                    #print(value_to_store)
                                    #print(stencil)
                                    step = 2
                                #print("prima " + str(self._edl))
                                self._edl.update({key : stencil})
                                #print("dopo " + str(self._edl))
                                #print(stencil)
                                #if (p_g_index == 148):
                                #    print(self._edl)
                                #    print(self._comm_w.Get_rank())
                        #if (p_g_index == 224):
                        #    print(stencil)

        # Columns indices for the \"PETSc\" matrix.
        c_indices = []

        is_ghost_inter = octree.get_is_ghost(inter,
                                             True)
        is_bound_inter = octree.get_bound(inter,
                                          0    ,
                                          True)
        # Normal to the intersection, and its numpy version.
        normal_inter, \
        n_normal_inter = octree.get_normal(inter,
                                           True) # We want also a \"numpy\"
                                                 # version
        # Normal's axis, indicating the non-zero component of the normal.
        n_axis = numpy.nonzero(n_normal_inter)[0][0]
        n_value = n_normal_inter[n_axis]
        # We are addding the indices of the interpolation done for the nodes of
        # the intersection, only if they are not on the background boundary.
        node_1_interpolated = True if (n_cs_n_is[1][0].size) else False
        node_0_interpolated = True if (n_cs_n_is[0][0].size) else False
        c_indices.extend(r_indices)
        if (grid and n_nodes_on_f_b):
            key_0 = grid                # Grid on which we are
            key_1 = r_indices[0]        # First index of the owners of the inter-
                                        # section
            key_2 = r_indices[1] if (n_nodes_on_f_b == 1) else \
                    n_cs_n_is[0][1][-2] # Or the second index of the owners of
                                        # the intersection, or the index of the
                                        # second to last of the neighbours of the
                                        # first node of the intersection (that,
                                        # if the intersection is not on the corner,
                                        # it will be inside the foregrounds; other-
                                        # wise, it will be equal to \"-1\".)
            key_3 = n_nodes_on_f_b      # How many nodes on the foreground boundary
                                        # for the current intersection
            key_4 = 0 if (n_nodes_on_f_b == 1) else \
                    n_cs_n_is[1][1][-2] # Or a useless field, or see the explication
                                        # for \"key_2\", but for the second node of
                                        # the intersection
            key_5 = n_axis              # Normal axis
            key_6 = n_value             # Value of the normal axis
            key_7 = rings[node_on_f_b][0] if (n_nodes_on_f_b == 1) else \
                    rings[0][0]
            key_8 = rings[node_on_f_b][1] if (n_nodes_on_f_b == 1) else \
                    rings[0][1]
            key_9 = rings[node_on_f_b][2] if (n_nodes_on_f_b == 1) else \
                    rings[0][2]
            key_10 = 0 if (n_nodes_on_f_b == 1) else \
                     rings[1][0]
            key_11 = 0 if (n_nodes_on_f_b == 1) else \
                     rings[1][1]
            key_12 = 0 if (n_nodes_on_f_b == 1) else \
                    rings[1][2]
            key = (key_0,
                   key_1,
                   key_2,
                   key_3,
                   key_4,
                   key_5,
                   key_6,
                   key_7,
                   key_8,
                   key_9,
                   key_10,
                   key_11,
                   key_12)
            l_stencil = 21 if (dimension == 2) else 31
            stencil = [-1] * l_stencil
            h = octree.get_area(inter        ,
                                is_ptr = True,
                                is_inter = True)
            # Only one node is on the foreground boundary...
            if (key_3 == 1):
                stencil[0] = h
                # Saving the coefficient of the corresponding node.
                coef_index = 2 if node_on_f_b else 3
                stencil[1] = n_coeffs[coef_index]
                j = 2
                # Saving the coordinates of the node in question.
                stencil[j : j + dimension] = \
                    nodes_inter[node_on_f_b][: dimension]
                j = j + dimension
                # Number of nodes in the ring (here, should be always equal to
                # 4).
                n_n_r =  n_cs_n_is[node_on_f_b][0].shape[0]
                for i in xrange(0, n_n_r):
                    # Storing all the coordinates of the neighbours in the ring
                    # of the node.
                    stencil[j : j + dimension] = \
                        n_cs_n_is[node_on_f_b][0][i][: dimension]
                    j = j + dimension
                # If the node on the boundary is the second (so, the node number
                # 1), we will have to compute new bilinear coefficients with the
                # octants on the background, so for the moment we will not store
                # the old interpolation...
                if (node_on_f_b):
                    node_1_interpolated = False
                # ...and the same speech is on also for the node 0.
                else:
                    node_0_interpolated = False
            # All the two of them are on the foreground boundaries.
            else:
                # Saving the size of the intersection because we will need it to
                # ri-evaluate, other than the bilinear coefficients, also the
                # interface coefficients.
                stencil[0] = h
                j = 1
                n_nodes = 2 if (dimension == 2) else 4
                for i in xrange(0, n_nodes):
                    # Saving coordinates of both nodes.
                    stencil[j : j + dimension] = \
                        nodes_inter[i][: dimension]
                    j = j + dimension
                # Saving all the coordinates of the rings of the nodes.
                for i in xrange(0, n_nodes):
                    # Number of nodes in the ring.
                    n_n_r =  n_cs_n_is[i][0].shape[0]
                    for k in xrange(0, n_n_r):
                        stencil[j : j + dimension] = n_cs_n_is[i][0][k][: dimension]
                        j = j + dimension
                # Old bilinear interpolation will not be used neither for the first
                # node, nor for the second.
                node_0_interpolated = False
                node_1_interpolated = False
                # the only index for the rows and for the columns will be filled
                # with \"-1\", to lets \"PETSc\" does nothing with it.
                # Using function \"update_bg_grids\", so no exact solution are
                # imposed in the "ghost" shell of the foreground, we have to keep
                # these following two lines decommented, because the function
                # \"update_bg_grids\" will do the same things of function
                # \"fill_mat\" for the octants of the foreground of the border,
                # in which we have to impose the coefficients for the octant
                # owner of the outer normal. Imposing the exact solution, in the
                # opposite way, will oblige us to comment these two lines, so in
                # this function \"fill_mat\# will be inserted the right coefficient
                # for the octant owner of the outer normal.
                r_indices = [-1] * len(r_indices)
                c_indices = [-1] * len(c_indices)
            #node_0_interpolated = False
            #node_1_interpolated = False

            self._edl.update({key : stencil})
        if (node_1_interpolated):
            c_indices.extend(n_cs_n_is[1][1])
        if (node_0_interpolated):
            c_indices.extend(n_cs_n_is[0][1])

        # Both the owners of the intersection are not penalized.
        if (len(r_indices) == 2):
            # Values to insert in \"r_indices\"; each sub list contains
            # values for each owner of the intersection.
            values = [[], []]
            # \"Numpy\" temporary array.
            n_t_array = numpy.array([n_coeffs[0],
                                     n_coeffs[1]])
            if (node_1_interpolated):
                n_t_array = numpy.append(n_t_array,
                                         coeffs_node_1)
            if (node_0_interpolated):
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
            # Here we can be only on the background, where some octants
            # are penalized.
            if (not is_bound_inter):
                if (node_1_interpolated):
                    n_t_array = numpy.append(n_t_array,
                                             coeffs_node_1)
                if (node_0_interpolated):
                    n_t_array = numpy.append(n_t_array,
                                             coeffs_node_0)
                mult = -1.0
                # Owner with the outer normal is not penalized, so we
                # have to add the coefficients in the corresponding row,
                # instead of subtract them.
                if (labels[0]):
                    mult = 1.0
                # Penalized global index, not masked.
                p_g_index = g_o_norms_inter[1 - labels[0]]
                # Not penalized global index, not masked.
                n_p_g_index = g_o_norms_inter[labels[0]]
                value_to_store = n_coeffs[1 - labels[0]] * mult

                m_n_p_g_index = mask_octant(n_p_g_index)
                #print(nodes_inter)
                # Setting exact solution for centers of owners of background intersections,
                # centers which are penalized.
                #nsolution = utilities.exact_sol(narray([[owners_centers[1 - labels[0]][0],
                #                                         owners_centers[1 - labels[0]][1]]]),
                #                                       b_alpha              ,
                #                                       b_beta               ,
                #                                       dim = 2              ,
                #                                       apply_mapping = True)
                #self._rhs.setValues(m_n_p_g_index,
                #                    value_to_store * -1.0 * nsolution[0],
                #                    insert_mode)
                #nsolution = utilities.exact_sol(narray([[nodes_inter[0][0],
                #                                         nodes_inter[0][1]]]),
                #                                       b_alpha              ,
                #                                       b_beta               ,
                #                                       dim = 2              ,
                #                                       apply_mapping = True)
                #self._rhs.setValues(m_n_p_g_index,
                #                    n_coeffs[3] * -1.0 * nsolution[0] * mult,
                #                    insert_mode)
                #nsolution = utilities.exact_sol(narray([[nodes_inter[1][0],
                #                                         nodes_inter[1][1]]]),
                #                                       b_alpha              ,
                #                                       b_beta               ,
                #                                       dim = 2              ,
                #                                       apply_mapping = True)
                #self._rhs.setValues(m_n_p_g_index,
                #                    n_coeffs[2] * -1.0 * nsolution[0] * mult,
                #                    insert_mode)


                key = (n_polygon + 1, \
                       p_g_index    , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0            , \
                       0)

                stencil = self._edl.get(key)
                displ = 1 + dimension
                step = 2
                l_stencil = 21 if (dimension == 2) else 31
                # Sometimes \"stencil\" is equal to \"None\" because
                # there are values of \"p_g_index\" which correspond to
                # ghost octant not included in the local octree, and in
                # the local \"self._edl\".
                # TODO: I think that with the add of the \"cur_proc_owner\" in
                #       the function \"fill_mat_and_rhs\", this \"if\" is no
                #       more useful.
                if (stencil):
                    for k in xrange(displ, l_stencil, step):
                        if (stencil[k] == n_p_g_index):
                            stencil[k + 1] = stencil[k + 1] + value_to_store
                            break
                else:
                    key = (n_polygon + 1, \
                           p_g_index    , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           0            , \
                           1)

                    stencil = self._edl.get(key)
                    displ = 1 + dimension
                    step = 2
                    l_stencil = 21 if (dimension == 2) else 31
                    if (stencil):
                        for k in xrange(displ, l_stencil, step):
                            if (stencil[k] == n_p_g_index):
                                stencil[k + 1] = stencil[k + 1] + value_to_store
                                break
                    #if (p_g_index == 639):
                    #    if (n_p_g_index == 725):
                    #        print("brlla")
            # We are on a boundary intersection; here normal is always
            # directed outside, so the owner is the one with the outer
            # normal.
            else:
                m_octant = m_g_o_norms_inter[labels[0]]
                mult = 1.0
                value_to_store = n_coeffs[1 - labels[0]] * mult
                if (is_background):
                    self.set_bg_b_c(inter         ,
                                    m_octant      ,
                                    owners_centers,
                                    n_normal_inter,
                                    labels        ,
                                    value_to_store)
                #else:
                #    key = (grid    ,
                #           m_octant,
                #           n_axis)
                #    stencil = self._edl.get(key)
                #    # The \"if\" clause is necessary because interface
                #    # could be on the boundary of the background, where
                #    # the exterior neighbour is not saved previously in
                #    # \"self._edl\" because it is outside the transfor-
                #    # med background.
                #    if (stencil):
                #        stencil[(2 * dimension) + 1] = value_to_store

            values = (n_t_array * mult).tolist()
        self._b_mat.setValues(r_indices, # Row
                              c_indices, # Columns
                              values   , # Values to be inserted
                              insert_mode)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def fill_rhs(self        ,
                 coeffs_sizes,   # How many coeffcients were evaluated for each
                                 # node of the intersection
                 labels      ,   # \"list\" containing indices for the owners
                                 # of the intersection, to know if they have
                                 # inneror outer normal
                 coeffs_nodes,   # Square method coefficients for intersection
                                 # nodes
                 r_indices   ,   # Matrix row indices
                 n_nodes_inter): # \"numpy\" coordinates of intersection nodes
        # Number of owners (of the intersection).
        n_owners = len(r_indices)
        dimension = self._dim
        # Number of nodes (of the intersection).
        n_nodes = 2 if (dimension == 2) else 4
        grid = self._proc_g
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        values_rhs = []
        # Number of values to add to the \"rhs\" for each intersection node (if
        # of course it is on the background boundary).
        n_values_rhs = 0

        insert_mode = PETSc.InsertMode.ADD_VALUES

        solution = utilities.exact_sol
        nsolution = lambda x : solution(x    ,
                                        alpha,
                                        beta ,
                                        dim = 2)
        # If the first octant owner of the intersection has the inner normal,
        # then the values should be subtracted, so added to the rhs. Viceversa
        # if the owner has the outgoing normal.
        mult = 1.0
        if (labels[0]):
            mult = -1.0

        e_sols = nsolution(n_nodes_inter)

        for i in xrange(0, n_nodes):
            # Number of \"i\" node coefficients.
            n_coeffs = coeffs_sizes[i]
            # The node \"i\" of the interface is not on the background boundary.
            if (n_coeffs):
                continue
            # The node \"i\" of the interface is on the background boundary.
            e_sol = e_sols[i]
            e_sol_coeff = coeffs_nodes[i]
            e_sol = mult * e_sol * e_sol_coeff
            values_rhs.append(e_sol)
            n_values_rhs = n_values_rhs + 1

        if (n_values_rhs):
            indices_rhs = [r_indices[0]] * n_values_rhs

            if (n_owners == 2):
                indices_rhs.extend([r_indices[1]] * n_values_rhs)
                tmp_values_rhs = [ -1.0 * value for value in values_rhs]
                values_rhs.extend(tmp_values_rhs)

            self._rhs.setValues(indices_rhs,
                                values_rhs ,
                                insert_mode)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Apply restriction/prolongation operators.
    def new_apply_rest_prol_ops(self,
                                row_indices,
                                col_indices,
                                col_values):
        """Method which applies the right coefficients at the right neighbours
           in the prolongaion and restriction blocks.

           Arguments:
                row_indices (list) : indices of the rows where to apply the
                                     coefficients.
                col_indices (list) : indices of the columns where to apply the
                                     coefficients.
                col_values (list) : elements to insert into \"col_indices\"."""
        insert_mode = PETSc.InsertMode.ADD_VALUES
        self._b_mat.setValues(row_indices,
                              col_indices,
                              col_values ,
                              insert_mode)

        msg = "Applied prolongation and restriction operators."
        self.log_msg(msg   ,
                     "info")

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
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        if (grid):
            is_background = False
            numpy_row_indices = numpy.array(row_indices)
            numpy_row_indices = numpy_row_indices[numpy_row_indices >= 0]
        insert_mode = PETSc.InsertMode.ADD_VALUES
        solution = utilities.exact_sol
        narray = numpy.array
        nsolution = lambda x : solution(narray([[x[0], x[1]]]),
                                               alpha          ,
                                               beta           ,
                                               dim = 2)
        n_rows = 1 if (is_background) else numpy_row_indices.size
        to_rhs = []
        # Exact solutions.
        e_sols = []

        for i, index in enumerate(col_indices):
            # If the neighbour is outside the background boundary, the exact
            # solution is evaluated.
            if (index == "outside_bg"):
                to_rhs.append(i)
                e_sol = nsolution((centers[i][0],
                                   centers[i][1]))
                e_sols.append(e_sol)

        for i in range(0, n_rows):
            row_index = row_indices if (is_background) else numpy_row_indices[i]
            co_indices = col_indices
            co_values = col_values
            if (not is_background):
                row_index = self._ngn[row_index]
            # If \"to_rhs\" is not empty.
            if (not not to_rhs):
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
                       solution      ,
                       h_s           ,
                       l2 = False    ,
                       # Return \"numpy\" data
                       r_n_d = False):
        """Function which evals the infinite and L2 norms of the error.

           Arguments:
                exact_solution (numpy.array) : exact solution.
                solution (numpy.array) : computed solution.
                h_s (numpy.array) : face's dimension for each octant

           Returns:
                (norm_inf, norm_L2) (tuple of int): evaluated norms."""

        numpy_difference = numpy.subtract(exact_solution,
                                          solution)
        norm_inf = numpy.linalg.norm(numpy_difference,
                                     # Type of norm we want to evaluate.
                                     numpy.inf)
        if (l2):
            norm_X2 = numpy.linalg.norm(numpy_difference,
                                        2)
            norm_X2_name = "l2"
        else:
            norm_X2 = numpy.linalg.norm(numpy_difference * h_s,
                                        2)
            norm_X2_name = "L2"

        msg = "Evaluated norms"
        extra_msg = "with (norm_inf, " + str(norm_X2_name) + ") = " + \
                    str((norm_inf, norm_X2))
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        if (r_n_d):
            return (numpy.array([norm_inf]),
                    numpy.array([norm_X2]))

        return (norm_inf, norm_X2)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def compute_function_on_nodes(self         ,
                                  inter        ,
                                  n_nodes_inter,
                                  n_cs_n_is    ,
                                  l_s_coeffs   ,
                                  # For the moment, for nodes of the foreground
                                  # grids which are on the foreground boundary,
                                  # we use the same approach used for the back-
                                  # ground ones, that is evaluating exact solution
                                  # on the boundary.
                                  # TODO: use also for the point on the foreground
                                  # boundaries the interpolation with the points
                                  # (correct and mapped on the background) found
                                  # later.
                                  are_nodes_on_f_b):
        octree = self._octree
        grid = self._proc_g
        alpha = self.get_trans(grid)[1]
        beta = self.get_trans(grid)[2]
        solution = utilities.exact_sol
        narray = numpy.array
        nsolution = lambda x : solution(x    ,
                                        alpha,
                                        beta ,
                                        dim = 2)
        e_nsolution_nodes = nsolution(n_nodes_inter)
        e_nsolution_node_0 = e_nsolution_nodes[0]
        e_nsolution_node_1 = e_nsolution_nodes[1]
        self._f_nodes_exact.append(e_nsolution_node_0)
        self._f_nodes_exact.append(e_nsolution_node_1)
        h_inter = octree.get_area(inter        ,
                                  is_ptr = True,
                                  is_inter = True)
        self._h_s_inter.append(h_inter)
        self._h_s_inter.append(h_inter)
        l_s_coeffs_s_0 = l_s_coeffs[0].shape[0]
        l_s_coeffs_s_1 = l_s_coeffs[1].shape[0]
        if ((l_s_coeffs[0].size == 0) or \
            (are_nodes_on_f_b[0] == True)):
            self._f_nodes.append(e_nsolution_node_0)
        else:
            f_s = nsolution(n_cs_n_is[0][0])
            f_0 = f_s[0]
            f_1 = f_s[1]
            f_2 = f_s[2]
            if (l_s_coeffs_s_0 == 4):
                f_3 = f_s[3]
            self._f_nodes.append(l_s_coeffs[0][0] * f_0 +
                                 l_s_coeffs[0][1] * f_1 +
                                 l_s_coeffs[0][2] * f_2 +
                                 ((l_s_coeffs[0][3] * f_3) if \
                                 (l_s_coeffs_s_0 == 4) else 0))
        if ((l_s_coeffs[1].size == 0) or \
            (are_nodes_on_f_b[1] == True)):
            self._f_nodes.append(e_nsolution_node_1)
        else:
            f_s = nsolution(n_cs_n_is[1][0])
            f_0 = f_s[0]
            f_1 = f_s[1]
            f_2 = f_s[2]
            if (l_s_coeffs_s_1 == 4):
                f_3 = f_s[3]
            self._f_nodes.append(l_s_coeffs[1][0] * f_0 +
                                 l_s_coeffs[1][1] * f_1 +
                                 l_s_coeffs[1][2] * f_2 +
                                 ((l_s_coeffs[1][3] * f_3) if \
                                 (l_s_coeffs_s_1 == 4) else 0))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def inter_coeffs(self ,
                     nodes,
                     point,
                     bil_inter = True):
        if (bil_inter):
            return utilities.bil_coeffs(nodes,
                                        point)
        return utilities.least_squares(nodes,
                                       point)
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
    def residual(self):
        return self._residual

    @property
    def sol(self):
        return self._sol

    @property
    def not_pen_centers(self):
        return self._centers_not_penalized

    @property
    def f_nodes(self):
        return numpy.array(self._f_nodes)

    @property
    def f_nodes_exact(self):
        return numpy.array(self._f_nodes_exact)

    @property
    def f_on_bord(self):
        return numpy.array(self._f_on_borders)

    @property
    def f_exact_on_bord(self):
        return numpy.array(self._f_on_borders_exact)

    @property
    def h_s_inter(self):
        return numpy.array(self._h_s_inter)

    @property
    def h_s_inter_on_board(self):
        return numpy.array(self._h_s_inter_on_board)

