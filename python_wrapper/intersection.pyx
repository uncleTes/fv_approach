from libc.stdint cimport uintptr_t

cdef extern from "Intersection.hpp" namespace "bitpit":
    cdef cppclass Intersection:
        Intersection() except +

        Intersection(Intersection& intersection) except +
    
        Intersection& operator=(Intersection&)

cdef class Py_Intersection:
    cdef Intersection* thisptr

    def __cinit__(self,
                  *args):
        cdef uintptr_t int_ptr = 0
        n_args = len(args)

        if (n_args == 0):
           self.thisptr = new Intersection()
        elif (n_args == 1):
            try:
                int_ptr = args[0]
                self.thisptr = new Intersection((<Intersection*><void*>int_ptr)[0])
            except Exception as e:
                traceback.print_exc()
        else:
            print("Dude, wrong number of input parameters.")    
    def __dealloc__(self):
        del self.thisptr

     
