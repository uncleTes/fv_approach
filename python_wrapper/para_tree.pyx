# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
include "octant.pyx"
include "intersection.pyx"

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
import os
import numpy
cimport numpy

IF (BITPIT_ENABLE_MPI):
    cimport mpi4py.MPI as MPI
    from mpi4py.libmpi cimport *
ELSE:
    #ctypedef struct _mpi_comm_t
    #ctypedef _mpi_comm_t* MPI_Comm
    ctypedef public int MPI_Comm

# Actually, this typedef is in "Map.hpp", but we just need it without the "Map"
# class so, here we are.
ctypedef public vector[darray3] darr3vector

#http://www.cplusplus.com/reference/bitset/bitset/
cdef extern from "<bitset>" namespace "std":
   cdef cppclass bitset[T]:
        bitset()
        # Type \"uint32_t\" is standard, \"uint32\" not.
        # http://stackoverflow.com/questions/13362084/difference-between-uint32-and-uint32-t
        bitset(uint32_t val) 
        bool operator[](size_t pos)
        size_t size()

cdef extern from *:
    ctypedef int _N72 "72"

cdef extern from "ParaTree.hpp" namespace "bitpit":
    ctypedef vector[bool] bvector
    ctypedef bitset[_N72] octantID
    ctypedef vector[Octant*] ptroctvector
    ctypedef vector[Octant*].iterator octantIterator

    cdef cppclass ParaTree:
        # MPI.
        ParaTree(uint8_t dim    ,
                 int8_t maxlevel,
                 string logfile , 
                 MPI_Comm mpi_comm) except +
        # No MPI.
        ParaTree(uint8_t dim    , 
                 int8_t maxlevel, 
                 string logfile) except +
        
        uint32_t getNumOctants() const

        void write(string filename)

        void writeTest(string filename, 
                       vector[double] data)
        
        darray3 getCenter(uint32_t idx)

        darray3 getCenter(Octant* octant)
	
        void computeConnectivity()

        #void computeGhostsConnectivity()

        bool adaptGlobalRefine(bool mapper_flag)
		
        bool adaptGlobalCoarse(bool mapper_flag)

        void updateConnectivity()

        #void updateGhostsConnectivity()

        void setBalance(uint32_t idx, 
                        bool balance)

        void setBalance(Octant* octant, 
                        bool balance)

        bool getBound(Octant* octant, 
                      uint8_t iface)

        uint64_t getGhostGlobalIdx(uint32_t idx)

        uint64_t getGlobalIdx(uint32_t idx)
        
        int findOwner(const uint64_t& morton)
        
        bool getIsNewC(uint32_t idx)

        bool getIsGhost(Octant* octant)

        Octant* getOctant(uint32_t idx)

        Octant* getGhostOctant(uint32_t idx) 

        void setMarker(Octant* octant, 
                       int8_t marker)

        void setMarker(uint32_t idx, 
                       int8_t marker)

        int8_t getMarker(Octant* octant)

        int8_t getMarker(uint32_t idx)

        bool adapt(bool mapper_flag)

        bool getBalance(Octant* octant)

        bool getBalance(uint32_t idx)

        #const u32arr3vector& getNodes()

        #const u32arr3vector& getGhostNodes()

        const u32vector2D& getConnectivity()

        const u32vector2D& getGhostConnectivity()

        uint32_t getNumNodes() const
        
        uint32_t getNumGhosts() const

        void loadBalance(dvector* weight)

        uint8_t getLevel(uint32_t idx)

        void findNeighbours(uint32_t idx,
                            uint8_t iface,
                            uint8_t codim,
                            u32vector& neighbours,
                            bvector& isghost)
        
        uint32_t getPointOwnerIdx(dvector& point)

        Octant* getPointOwner(darray3& point)

        Octant* getPointOwner(dvector& point)
	
        uint64_t getGlobalNumOctants()
	
        uint8_t getNfaces()
        
        uint8_t getNnodes()

        void getFacenode(uint8_t nodeface[6][4])

        # FV approach...
        uint32_t getNumIntersections()

        Intersection* getIntersection(uint32_t idx)

        u32vector getOwners(Intersection* inter)

        uint8_t getFace(Intersection* inter)

        bool getBound(Intersection* inter)

        bool getFiner(Intersection* inter)

        darr3vector getNodes(Intersection* inter)
        darr3vector getNodes(Octant* idx)
        darr3vector getNodes(uint32_t idx)

        double getArea(Intersection* inter)

        darray3 getNormal(Intersection* inter)

        bool getIsGhost(Intersection* inter)

        uint8_t getLevel(Octant* octant)

        void computeIntersections()

cdef class Py_Para_Tree:
    cdef ParaTree* thisptr
    cdef MPI_Comm mpi_comm

    def __cinit__(self, 
                  *args):
        # To derive \#Py_Para_Tree\", I think is better set the \"thisptr\" as
        # \"NULL\" instead of doing in the derived class what is done in the 
        # following link:
        # http://stackoverflow.com/questions/28573479/cython-python-c-inheritance-passing-derived-class-as-argument-to-function-e
        # that could be followed instead for the \"__dealloc__\" method.
        thisptr = NULL

        if (type(self) is Py_Para_Tree):
            n_args = len(args)
	    	
            if (n_args == 0):
                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.thisptr = new ParaTree(2          , 
                                                20         , 
                                                "PABLO.log", 
                                                mpi_comm)
                ELSE:
                    self.thisptr = new ParaTree(2 , 
                                                20, 
                                                "PABLO.log")
            elif (n_args == 1):
                dim = args[0]

                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.thisptr = new ParaTree(dim        , 
                                                20         , 
                                                "PABLO.log", 
                                                mpi_comm)
                ELSE:
                    self.thisptr = new ParaTree(dim, 
                                                20 , 
                                                "PABLO.log")
            elif (n_args == 2):
                dim = args[0]
                max_level = args[1]

                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.thisptr = new ParaTree(dim        , 
                                                max_level  , 
                                                "PABLO.log", 
                                                mpi_comm)
                ELSE:
                    self.thisptr = new ParaTree(dim      , 
                                                max_level, 
                                                "PABLO.log")
            elif (n_args == 3):
                dim = args[0]
                max_level = args[1]
                log_file = args[2]
                
                IF (BITPIT_ENABLE_MPI):
                    mpi_comm = MPI_COMM_WORLD
                    self.thisptr = new ParaTree(dim      ,
                                                max_level,
                                                log_file ,
                                                mpi_comm)
                ELSE:
                    self.thisptr = new ParaTree(dim      ,
                                                max_level,
                                                log_file)
            else:
                IF (BITPIT_ENABLE_MPI):
                    if (n_args == 4):
                        dim = args[0]
                        max_level = args[1]
                        log_file = args[2]
                        mpi_comm = (<MPI.Comm>args[3]).ob_mpi

                        self.thisptr = new ParaTree(dim      ,
                                                    max_level,
                                                    log_file ,
                                                    mpi_comm)
                ELSE:
                    print("Dude, wrong number of input arguments. Type " +
                          "\"help(Py_Para_Tree)\".")
	
    def __dealloc__(self):
        # Why delete a possible \"NULL\" pointer?
        # http://stackoverflow.com/questions/615355/is-there-any-reason-to-check-for-a-null-pointer-before-deleting
        if (type(self) is Py_Para_Tree):
            del self.thisptr
	
    def get_num_octants(self):
        return self.thisptr.getNumOctants()
	
    def write(self, 
              string file_name):
        self.thisptr.write(file_name)

    def write_test(self            , 
                   string file_name, 
                   vector[double] data):
        cdef string c_file_name = file_name
        cdef vector[double] c_data = data
        cdef darray3 c_array

        self.thisptr.writeTest(c_file_name, c_data)

    def get_center(self         , 
                   uintptr_t idx, 
                   ptr_octant = False):
        cdef darray3 center
        py_center = []
        
        if (ptr_octant):
            center = self.thisptr.getCenter(<Octant*><void*>idx)
        else:
            center = self.thisptr.getCenter(<uint32_t>idx)
        
        for i in xrange(0, center.size()):
            py_center.append(center[i])
        
        return py_center
	
    def compute_connectivity(self):
        self.thisptr.computeConnectivity()

    #def compute_ghosts_connectivity(self):
    #    self.thisptr.computeGhostsConnectivity()

    def adapt_global_refine(self,
                            bool mapper_flag = False):
        return self.thisptr.adaptGlobalRefine(mapper_flag)
	
    def adapt_global_coarse(self,
                            bool mapper_flag = False):
        return self.thisptr.adaptGlobalCoarse(mapper_flag)
	
    def update_connectivity(self):
        self.thisptr.updateConnectivity()

    #def update_ghosts_connectivity(self):
    #    self.thisptr.updateGhostsConnectivity()

    def set_balance(self         , 
                    uintptr_t idx, 
                    bool balance ,
                    ptr_octant = False):
        cdef Octant* octant = NULL

        if (ptr_octant):
            self.thisptr.setBalance(<Octant*><void*>octant, 
                                    balance)
        else:
            self.thisptr.setBalance(<uint32_t>idx, 
                                    balance)

    def get_bound(self            , 
                  uintptr_t octant, 
                  uint8_t iface   ,
	          bool is_ptr_inter = False):
        if (is_ptr_inter):
            return self.thisptr.getBound(<Intersection*><void*>octant)
        else:
            return self.thisptr.getBound(<Octant*><void*>octant,
                                         iface)

    def get_ghost_global_idx(self, 
                             uint32_t idx):
        return self.thisptr.getGhostGlobalIdx(idx)

    def get_global_idx(self, uint32_t idx):
        return self.thisptr.getGlobalIdx(idx)
    
    def find_owner(self, 
                   uint64_t& g_index):
        return self.thisptr.findOwner(g_index)

    def get_is_new_c(self, 
                     uint32_t idx):
        return self.thisptr.getIsNewC(idx)
        
    def get_is_ghost(self         ,
                     uintptr_t ptr,
                     # By default, \"ptr\" will be treated as a pointer to an
                     # \"Octant\", and not to an \"Intersection\". Of course the
                     # user has to choose between them.
                     is_ptr_inter = False):
        if (not is_ptr_inter):
            return self.thisptr.getIsGhost(<Octant*><void*>ptr)
        else:
            return self.thisptr.getIsGhost(<Intersection*><void*>ptr)
    def get_octant(self, 
                   uint32_t idx):
        cdef Octant* octant
        
        octant = self.thisptr.getOctant(idx)
        py_oct = <uintptr_t>octant

        return py_oct
	
    def get_ghost_octant(self, 
                         uint32_t idx):
        cdef Octant* octant
        
        octant = self.thisptr.getGhostOctant(idx)
        py_oct = <uintptr_t>octant
        
        return py_oct

    def set_marker(self            , 
                   uintptr_t octant, 
                   int8_t marker   , 
                   ptr_octant = False):
        if (ptr_octant):
            self.thisptr.setMarker(<Octant*><void*>octant, 
                                   marker)
        else:
            self.thisptr.setMarker(<uint32_t>octant,
                                   marker)

    def get_marker(self            , 
                   uintptr_t octant, 
                   ptr_octant = False):
        if (ptr_octant):
            return self.thisptr.getMarker(<Octant*><void*>octant)
            
        return self.thisptr.getMarker(<uint32_t>octant)
    
    def adapt(self,
              bool mapper_flag = False):
        return self.thisptr.adapt(mapper_flag)

    def get_balance(self            , 
                    uintptr_t octant,
                    ptr_octant = False):
        if (ptr_octant):
            return self.thisptr.getBalance(<Octant*><void*>octant)
            
        return self.thisptr.getBalance(<uint32_t>octant)
        
    #def get_nodes(self):
    #    cdef u32arr3vector nodes
    #    cdef u32array3 array
    #    cdef int v_size
    #    cdef int i
    #    cdef int j
    #    cdef int a_size = 3
    #    nodes = self.thisptr.getNodes()
    #    v_size = nodes.size()
    #    py_nodes = []

    #    for i in xrange(0, v_size):
    #        array = nodes[i]
    #        py_array = []
    #        for j in xrange(0, a_size):
    #            py_array.append(array[j])
    #        py_nodes.append(py_array)

    #    return py_nodes
    
    #def get_ghost_nodes(self):
    #    cdef u32arr3vector nodes
    #    cdef u32array3 array
    #    cdef int v_size
    #    cdef int i
    #    cdef int j
    #    cdef int a_size = 3
    #    nodes = self.thisptr.getGhostNodes()
    #    v_size = nodes.size()
    #    py_nodes = []

    #    for i in xrange(0, v_size):
    #        array = nodes[i]
    #        py_array = []
    #        for j in xrange(0, a_size):
    #            py_array.append(array[j])
    #        py_nodes.append(py_array)

    #    return py_nodes

    def get_connectivity(self):
        return self.thisptr.getConnectivity()

    def get_ghost_connectivity(self):
        return self.thisptr.getGhostConnectivity()

    def get_num_nodes(self):
        return self.thisptr.getNumNodes()
	
    def get_num_ghosts(self):
        return self.thisptr.getNumGhosts()

    IF (BITPIT_ENABLE_MPI):
        def load_balance(self, 
                         *args):
            cdef dvector* weight = NULL
            n_args = len(args)
            
            if (n_args == 1):
                weight = <dvector*><void*>args[0]

            self.thisptr.loadBalance(weight)
    
    def get_level(self            ,
                  uintptr_t octant,
                  ptr_octant = False):
        if (ptr_octant):
            return self.thisptr.getLevel(<Octant*><void*>octant)

        return self.thisptr.getLevel(<uint32_t>octant)

    def find_neighbours(self                 , 
                        uint32_t idx         ,
                        uint8_t iface        ,
                        uint8_t codim        ,
                        u32vector& neighbours,
                        bvector& isghost):
        self.thisptr.findNeighbours(idx, 
                                    iface, 
                                    codim, 
                                    neighbours,
				    isghost)
        
        return (neighbours, isghost)

    def get_point_owner_idx(self, 
                            dvector& point):
        return self.thisptr.getPointOwnerIdx(<dvector&>point)

    def get_point_owner(self ,
                        point,
                        is_array = False):
        cdef Octant* octant

        if (is_array):
            octant = self.thisptr.getPointOwner(<darray3&>point)
        else:
            octant = self.thisptr.getPointOwner(<dvector&>point)

        py_oct = <uintptr_t>octant

        return py_oct

    def get_global_num_octants(self):
        return self.thisptr.getGlobalNumOctants()

    def get_n_faces(self):
        return self.thisptr.getNfaces()
    
    def get_n_nodes(self):
        return self.thisptr.getNnodes()

    def get_face_node(self):
        cdef numpy.ndarray[dtype=numpy.uint8_t, mode="c", ndim=2] face_node = numpy.zeros((6, 4), dtype = numpy.uint8)

        self.thisptr.getFacenode(<uint8_t (*)[4]>&face_node[0, 0])

        return face_node.tolist()

    # FV approach...
    def get_num_intersections(self):
        return self.thisptr.getNumIntersections()

    def get_intersection(self,
                         uint32_t idx):
        cdef Intersection* inter

        inter = self.thisptr.getIntersection(idx)
        py_inter = <uintptr_t>inter

        return py_inter

    def get_owners(self,
                   uintptr_t inter):
        return self.thisptr.getOwners(<Intersection*><void*>inter)

    def get_face(self,
                 uintptr_t inter):
        return self.thisptr.getFace(<Intersection*><void*>inter)

    #def get_bound(self,
    #              uintptr_t inter):
    #    return self.thisptr.getBound(<Intersection*><void*>inter)

    def get_finer(self,
                  uintptr_t inter):
        return self.thisptr.getFiner(<Intersection*><void*>inter)

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
                nodes = self.thisptr.getNodes(<Intersection*><void*>idx)
            else:
                nodes = self.thisptr.getNodes(<Octant*><void*>idx)
        else:
            nodes = self.thisptr.getNodes(<uint32_t>idx)
        for i in xrange(0, v_size):
            py_node = []
            # We are doing this second \"for\" loop because without it, Cython
            # compiler would give us the following error:
            # \"Cannot convert 'darray3' to Python object\".
            for j in xrange(0, a_size):
                py_node.append(nodes[i][j])

            py_nodes.append(py_node)

        return py_nodes

    def get_area(self,
                 uintptr_t inter):
        area = self.thisptr.getArea(<Intersection*><void*>inter)

        return area

    def get_normal(self,
                   uintptr_t inter):
        cdef darray3 normal
        # Size of the \"normal\".
        cdef int n_size = 3
        cdef int i
        py_normal = []

        normal = self.thisptr.getNormal(<Intersection*><void*>inter)

        for i in xrange(0, n_size):
            py_normal.append(normal[i])

        return py_normal

    def compute_intersections(self):
       self.thisptr.computeIntersections() 
