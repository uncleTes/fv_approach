# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
include "para_tree.pyx"

cdef extern from *:
    ctypedef int _N3 "3"

cdef extern from "MyPabloUniform.hpp" namespace "bitpit":
    ctypedef vector[darray3] darray3vector

    cdef cppclass MyPabloUniform(ParaTree):
        MyPabloUniform(uint8_t dim    , 
                       int8_t maxlevel, 
                       string logfile , 
                       MPI_Comm comm) except +

        MyPabloUniform(double X       , 
                       double Y       , 
                       double Z       , 
                       double L       , 
                       uint8_t dim    , 
                       int8_t maxlevel, 
                       string logfile , 
                       MPI_Comm comm) except +
        
        MyPabloUniform(uint8_t dim    , 
                       int8_t maxlevel, 
                       string logfile) except +

        MyPabloUniform(double X       , 
                       double Y       , 
                       double Z       , 
                       double L       , 
                       uint8_t dim    , 
                       int8_t maxlevel, 
                       string logfile) except +
        
        darray3 _getCenter(uint32_t idx)

        darray3 _getCenter(Octant* octant)
        
        uint32_t _getPointOwnerIdx(darray3& point)

        Octant* _getPointOwner(darray3& point)

        darray3 _getGhostNodeCoordinates(uint32_t inode)
        darray3 _getNodeCoordinates(uint32_t inode)

        # FV approach...
        double _getArea(Intersection* inter)

        darray3 _getNormal(Intersection* inter)
        
        darr3vector _getNodes(Intersection* inter)
        darr3vector _getNodes(Octant* idx)
        darr3vector _getNodes(uint32_t idx)

cdef class Py_My_Pablo_Uniform(Py_Para_Tree):
    cdef MyPabloUniform* der_thisptr

    def __cinit__(self, 
                  *args):
        der_thisptr = NULL

        if (type(self) is Py_My_Pablo_Uniform):
            n_args = len(args)
	    	
            if (n_args == 0):
                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.der_thisptr = new MyPabloUniform(2          , 
                                                          20         , 
                                                          "PABLO.log", 
                                                          mpi_comm)
                ELSE:
                    self.der_thisptr = new MyPabloUniform(2 , 
                                                          20, 
                                                          "PABLO.log")
            elif (n_args == 1):
                dim = args[0]

                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.der_thisptr = new MyPabloUniform(dim        , 
                                                          20         , 
                                                          "PABLO.log", 
                                                          mpi_comm)
                ELSE:
                    self.der_thisptr = new MyPabloUniform(dim, 
                                                          20 , 
                                                          "PABLO.log")
            elif (n_args == 2):
                dim = args[0]
                max_level = args[1]

                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.der_thisptr = new MyPabloUniform(dim        , 
                                                          max_level  , 
                                                          "PABLO.log", 
                                                          mpi_comm)
                ELSE:
                    self.der_thisptr = new MyPabloUniform(dim      , 
                                                          max_level, 
                                                          "PABLO.log")
            elif (n_args == 3):
                dim = args[0]
                max_level = args[1]
                log_file = args[2]
                
                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.der_thisptr = new MyPabloUniform(dim      ,
                                                          max_level,
                                                          log_file ,
                                                          mpi_comm)
                ELSE:
                    self.der_thisptr = new MyPabloUniform(dim      ,
                                                          max_level,
                                                          log_file)
            elif (n_args == 4):
                IF (BITPIT_ENABLE_MPI):
                    dim = args[0]
                    max_level = args[1]
                    log_file = args[2]
                    mpi_comm = (<MPI.Comm>args[3]).ob_mpi

                    self.der_thisptr = new MyPabloUniform(dim      ,
                                                          max_level,
                                                          log_file ,
                                                          mpi_comm)
                ELSE:
                    print("Dude, wrong number of input arguments. Type " +
                          "\"help(Py_Para_Tree)\".")
            elif (n_args == 7):
                x = args[0]
                y = args[1]
                z = args[2]
                l = args[3]
                dim = args[4]
                max_level = args[5]
                log_file = args[6]

                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.der_thisptr = new MyPabloUniform(x,
                                                          y,
                                                          z,
                                                          l,
                                                          dim,
                                                          max_level,
                                                          log_file,
                                                          mpi_comm)
                ELSE:
                    self.der_thisptr = new MyPabloUniform(x,
                                                          y,
                                                          z,
                                                          l,
                                                          dim,
                                                          max_level,
                                                          log_file)
            elif (n_args == 8):
                IF (BITPIT_ENABLE_MPI):
                    x = args[0]
                    y = args[1]
                    z = args[2]
                    l = args[3]
                    dim = args[4]
                    max_level = args[5]
                    log_file = args[6]
                    mpi_comm = (<MPI.Comm>args[7]).ob_mpi
                    self.der_thisptr = new MyPabloUniform(x,
                                                          y,
                                                          z,
                                                          l,
                                                          dim,
                                                          max_level,
                                                          log_file,
                                                          mpi_comm)
                ELSE:
                    print("Dude, wrong number of input arguments. Type " +
                          "\"help(Py_Para_Tree)\".")

            self.thisptr = self.der_thisptr
                    
    def __dealloc__(self):
        if (type(self) is Py_My_Pablo_Uniform):
            # This assignment prevents from the following compile error:
            # \"Error in `python': double free or corruption (!prev):\"
            # http://stackoverflow.com/questions/14063791/double-free-or-corruption-but-why
            self.thisptr = NULL
            del self.der_thisptr
            del self.thisptr

    def get_center(self         , 
                   uintptr_t idx, 
                   ptr_octant = False):
        cdef darray3 center
        py_center = []
        
        if (ptr_octant):
            center = self.der_thisptr._getCenter(<Octant*><void*>idx)
        else:
            center = self.der_thisptr._getCenter(<uint32_t>idx)
        
        for i in xrange(0, center.size()):
            py_center.append(center[i])
        
        return py_center
    
    def get_point_owner_idx(self, 
                            point):
        cdef darray3 ar_point
        n_coordinates = len(point)

        for i in xrange(0, n_coordinates):
            ar_point[i] = point[i]

        return self.der_thisptr._getPointOwnerIdx(ar_point)
        # The \"return self.der_thisptr._getPointOwnerIdx(<darray3&>ar_point)\"
        # call instead of the previous full body of the function
        # \"get_point_owner_idx\" would return a wrong value. I think that
        # is necessary an explicit conversion from python objects and 
        # \"darray3\", thing that in the previous version of \"PABLitO\" was not
        # necessary, being the input type of \"getPointOwnerIdx\" a \"dvector\".

    def get_point_owner(self,
                        point):
        cdef Octant* octant

        octant = self.der_thisptr._getPointOwner(<darray3&>point)

        py_oct = <uintptr_t>octant

        return py_oct

    def apply_persp_trans(self      ,
                          dimension , 
                          p_t_coeffs, 
                          logger    , 
                          log_file):
        import utilities
        # Number of nodes.
        cdef size_t n_nodes = self.thisptr.getNumNodes()
        # Number of ghost nodes.
        #cdef size_t n_g_nodes = (self.thisptr.getGhostNodes()).size()
        cdef int index
        cdef darray3 coordinates
        py_coordinates = [0.0] * 3
        # Geo nodes.
        g_nodes = []
        # Ghost geo nodes.
        g_g_nodes = []

        for index in xrange(0, n_nodes):
            coordinates = self.der_thisptr._getNodeCoordinates(index)
            for i in xrange(0, dimension):
                py_coordinates[i] = coordinates[i]

            to_append = utilities.apply_persp_trans(dimension     ,
                                                    py_coordinates, 
                                                    p_t_coeffs    , 
                                                    logger        , 
                                                    log_file)
            g_nodes.append(to_append)
        
        #for index in xrange(0, n_g_nodes):
        #    coordinates = self.der_thisptr._getGhostNodeCoordinates(index)
        #    for i in xrange(0, dimension):
        #        py_coordinates[i] = coordinates[i]
        #    
        #    to_append = utilities.apply_persp_trans(dimension     ,
        #                                            py_coordinates, 
        #                                            p_t_coeffs    , 
        #                                            logger        , 
        #                                            log_file)
        #    g_g_nodes.append(to_append)

        return g_nodes

    # FV approach...
    def get_area(self,
                 uintptr_t inter):

        area = self.der_thisptr._getArea(<Intersection*><void*>inter)

        return area

    def get_normal(self,
                   uintptr_t inter):
        cdef darray3 normal
        # Size of the \"normal\".
        cdef int n_size = 3
        cdef int i
        py_normal = []

        normal = self.der_thisptr._getNormal(<Intersection*><void*>inter)

        for i in xrange(0, n_size):
            py_normal.append(normal[i])

        return py_normal
    
    def get_nodes(self               ,
                  uintptr_t idx      ,
                  int dim            ,
                  bool is_ptr = False,
                  bool is_inter = False):
        cdef darr3vector nodes
        #cdef darray3 node
        # Vector size.
        cdef int v_size = 4 if (dim == 2) else 8
        # Array size.
        cdef int a_size = 3
        cdef int i
        cdef int j
        py_nodes = []

        if (is_ptr):
            if (is_inter):
                nodes = self.der_thisptr._getNodes(<Intersection*><void*>idx)
            else:
                nodes = self.der_thisptr._getNodes(<Octant*><void*>idx)
        else:
            nodes = self.der_thisptr._getNodes(<uint32_t>idx)
        for i in xrange(0, v_size):
            py_node = []
            # We are doing this second \"for\" loop because without it, Cython
            # compiler would give us the following error:
            # \"Cannot convert 'darray3' to Python object\".
            for j in xrange(0, a_size):
                py_node.append(nodes[i][j])

            py_nodes.append(py_node)

        #print(py_nodes)

        return py_nodes
