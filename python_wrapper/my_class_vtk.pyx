# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
import numpy as np
cimport numpy as np

include "my_pablo_uniform.pyx"

cdef extern from *:
    ctypedef int D2 "2"

cdef extern from "My_Class_VTK.hpp" namespace "bitpit":
    cdef cppclass My_Class_VTK[G, D, dim]:
        My_Class_VTK(D* data_    ,
                     G& grid_    ,
                     string dir_ , 
                     string name_, 
                     string cod_ , 
                     int ncell_  , 
                     int npoints_, 
                     int nconn_) except +

        void printVTK()

        void AddData(string name_, 
                     int comp_   , 
                     string type_, 
                     string loc_ , 
                     string cod_)

        void applyTransf(vector[darray3]& transGeoNodes)


cdef class Py_My_Class_VTK:
    cdef My_Class_VTK[MyPabloUniform,
                      double        ,
                      D2]* thisptr

    def __cinit__(self                                         , 
                  np.ndarray[double, ndim = 2, mode = "c"] data,
                  octree                                       ,
                  string directory                             ,
                  string file_name                             ,
                  string file_type                             ,
                  int n_cells                                  ,
                  int n_points                                 ,
                  int n_conn):
        self.thisptr = new My_Class_VTK[MyPabloUniform,
                                        double        ,
                                        D2](&data[0, 0]                                 ,
                                            (<Py_My_Pablo_Uniform>octree).der_thisptr[0],
                                            directory                                   ,
                                            file_name                                   ,
                                            file_type                                   ,
                                            n_cells                                     ,
                                            n_points                                    ,
                                            n_conn)

    def __dealloc__(self):
        del self.thisptr

    def print_vtk(self):
        self.thisptr.printVTK()

    def add_data(self              ,
                 string dataName   ,
                 int dataDim       ,
                 string dataType   ,
                 string pointOrCell,
                 string fileType):
        self.thisptr.AddData(dataName   ,
                             dataDim    ,
                             dataType   ,
                             pointOrCell,
                             fileType)
    
    def apply_trans(self,
                    geo_nodes):
        cdef vector[darray3] C_geo_nodes
        cdef vector[darray3] C_ghost_geo_nodes
        cdef darray3 node 

        for geo_node in geo_nodes:
            for i in xrange(0, len(geo_node)):
                node[i] = geo_node[i]
            C_geo_nodes.push_back(node)
        
        #for ghost_geo_node in geo_ghost_nodes:
        #    for i in xrange(0, len(ghost_geo_node)):
        #        node[i] = ghost_geo_node[i]
        #    C_ghost_geo_nodes.push_back(node)

                
        #self.thisptr.applyTransf(<vector[darray3]&>C_geo_nodes, <vector[darray3]&>C_ghost_geo_nodes)
        self.thisptr.applyTransf(<vector[darray3]&>C_geo_nodes)

