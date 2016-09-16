# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int8_t, uintptr_t
import traceback

# Using \"std::array\":
# https://gist.github.com/pierriko/fca8eebf49fca8336c50
# There is also \#array\" from cython:
# https://github.com/cython/cython/blob/master/Cython/Includes/cpython/array.pxd
cdef extern from "<array>" namespace "std":
    cdef cppclass array[T, N]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        array()
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        size_t size()

cdef extern from *:
    ctypedef int _N3 "3"

ctypedef public vector[uint8_t] u8vector
ctypedef public vector[uint32_t] u32vector
ctypedef public vector[uint64_t] u64vector
ctypedef public vector[double] dvector
ctypedef public array[double, _N3] darray3
ctypedef public array[int8_t, _N3] i8array3
ctypedef public array[uint32_t, _N3] u32array3
ctypedef public vector[vector[uint32_t]] u32vector2D
ctypedef public vector[vector[uint64_t]] u64vector2D
ctypedef public vector[u32array3] u32arr3vector

cdef extern from "Octant.hpp" namespace "bitpit":
    cdef cppclass Octant:
        #Octant() except +
        Octant(Octant& octant) except +

cdef class Py_Octant:
    """
    Returns an \"Octant\" object usable in Python. Its constructor accepts a
    number of input parameters included between 0 an 2:
    
    0)

    >>> oct = Py_Octant()

    1)

    >>> oct00 = Py_Octant()
    >>> oct01 = Py_Octant(oct00)

    2) This last case needs an integer as first parameter, because it is thought
       to be used by \"Py_Para_Tree\" for some of its methods, which need a 
       reference or a pointer to an \"Octant\" object.
       The second paramater is a boolean; if \"False\", an \"AssertionError\"
       will be throw, and nothing is built.

    >>> oct = Py_Octant(uitnptr_t, True)

    Args:
        oct (Optional[Py_Oct, uitnptr_t]): reference to another \"Oy_Oct\" or
                                           \"Octant\".
        is_ptr_oct (Optional[bool]): confirms that the previous parameter is a
                                     reference to an \"Octant\" object.
    
    Attributes:
        thisptr (Octant*) : pointer to the C++ class \"Octant\".

    """
    cdef Octant* thisptr

    def __cinit__(self, 
                  *args):
        # \"uintptr_t\" is an unsigned int capable of storing a pointer:
        # http://stackoverflow.com/questions/1845482/what-is-uintptr-t-data-type
        cdef uintptr_t oct_ptr = 0
        n_args = len(args)

        #if (n_args == 0):
        #    self.thisptr = new Octant()
        #elif (n_args == 1):
        if (n_args == 1):
            try:
                self.thisptr = new Octant((<Py_Octant>args[0]).thisptr[0])
            except Exception as e:
                traceback.print_exc()
        # http://stackoverflow.com/questions/22435992/python-trying-to-place-keyword-arguments-after-args
        elif (n_args == 2):
            try:
                is_ptr_oct = args[1]
                assert(is_ptr_oct is True)
                oct_ptr = args[0]
                self.thisptr = new Octant((<Octant*><void*>oct_ptr)[0])
            except Exception as e:
                traceback.print_exc()
        else:
            print("Dude, wrong number of input parameters. Type " +
                  "\"help(Py_Octant)\".")    

    def __dealloc__(self):
        del self.thisptr
