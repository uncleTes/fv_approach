# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import xml.etree.cElementTree as ET
import logging
import os
import sys
from mpi4py import MPI
from libcpp cimport bool
import numpy
cimport numpy
import math
# ------------------------------------------------------------------------------

# ----------------------------------FUNCTIONS-----------------------------------
# Suppose you have the string str = \"0, 1, 0\", calling this function as
# \"get_list_from_string(str, ", ", False)\" will return the list 
# \"[0.0, 1.0, 0.0]\".
#http://stackoverflow.com/questions/19334374/python-converting-a-string-of-numbers-into-a-list-of-int
def get_list_from_string(string  , 
                         splitter, 
                         integer = True):
    try:
        assert isinstance(string,
                          basestring)
        return [int(number) if integer else float(number) 
                for number in string.split(splitter)]
    except AssertionError:
        print("Parameter " + str(string) + " is not  an instance of " +
              "\"basestring\"")
        return None

# Suppose you have the string str = \"0, 1, 0; 1.5, 2, 3\", calling this
# function as \"get_lists_from_string(str, "; ", ", "False)\" will return the
# list \"[[0.0, 1.0, 0.0], [1.5, 2, 3]]\".
def get_lists_from_string(string            , 
                          splitter_for_lists, 
                          splitter_for_list , 
                          integer = False):
    try:
        assert isinstance(string,
                          basestring)
        return [get_list_from_string(string_chunk     , 
                                     splitter_for_list, 
                                     integer) 
                for string_chunk in string.split(splitter_for_lists)
               ]    
    except AssertionError:
        print("Parameter " + str(string) + " is not  an instance of " +
              "\"basestring\"")
        return None

# Suppose you have the list \"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\"; calling 
# this function as \"chunk_list(lst, 3)\", will return the following list:
# \"[[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]\".
# http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
def chunk_list(list_to_chunk, 
               how_many_parts):
    return [list_to_chunk[i::how_many_parts] for i in xrange(how_many_parts)]

def chunk_list_ordered(l_to_chunk, 
                       n_grids):
    # List length.
    l_l = len(l_to_chunk)
    # Size of normal sublist.
    s = l_l / n_grids
    # Extra elements for the last sublist.
    e_els = l_l - (n_grids * s)
    # Returned list.
    r_l = []
    # End first chunk.
    e_f_c = s + e_els
    r_l.append(l_to_chunk[0 : e_f_c])
    for i in range(0, n_grids - 1):
        r_l.append(l_to_chunk[(e_f_c + (s * i)) : (e_f_c + (s * i) + s)])

    return r_l

def get_proc_grid(l_lists,
                  w_rank):
    for i, l_list in enumerate(l_lists):
        for j, rank in enumerate(l_list):
            if w_rank == rank:
                return i

    return None

def split_list_in_two(list_to_be_splitted):
    half_len = (len(list_to_be_splitted) / 2)

    return list_to_be_splitted[:half_len], list_to_be_splitted[half_len:]

#def write_vtk_multi_block_data_set(**kwargs):
def write_vtk_multi_block_data_set(kwargs = {}):
    file_name = kwargs["file_name"]
    directory_name = kwargs["directory"]

    VTKFile = ET.Element("VTKFile"                    , 
                         type = "vtkMultiBlockDataSet", 
                         version = "1.0"              ,
                         byte_order = "LittleEndian"  ,
                         compressor = "vtkZLibDataCompressor")

    vtkMultiBlockDataSet = ET.SubElement(VTKFile, 
                                         "vtkMultiBlockDataSet")

    iter = 0
    for pablo_file in kwargs["pablo_file_names"]:
        for vtu_file in kwargs["vtu_files"]:
            if pablo_file in vtu_file:
                DataSet = ET.SubElement(vtkMultiBlockDataSet,
                                        "DataSet"           ,
                                        group = str(iter)   ,
                                        dataset = "0"       ,
                                        file = vtu_file)
                
        iter += 1

    vtkTree = ET.ElementTree(VTKFile)
    file_to_write = directory_name + str("/") + file_name
    vtkTree.write(file_to_write)

def check_null_logger(logger, 
                      log_file):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def log_msg(message , 
            log_file, 
            logger = None):
    logger = check_null_logger(logger, log_file)
    logger.info(message.center(140, "-"))
    return logger

def find_files_in_dir(extension, 
                      directory):
    files_founded = []

    for file in os.listdir(directory):
        if file.endswith(extension):
            files_founded.append(file)

    return files_founded

def set_class_logger(obj     , 
                     log_file,
                     what_log):
    obj_logger = Logger(type(obj).__name__,
                        log_file          ,
                        what_log).logger
    return obj_logger

def check_mpi_intracomm(comm  , 
                        logger,
                        type = "local"):
    if isinstance(comm, MPI.Intracomm):
        l_comm = comm
        logger.info("Setted "                                   +
                    ("local " if type == "local" else "world ") +
                    "comm \""                                   +
                    str(comm.Get_name())                        + 
                    "\" and rank \""                            +
                    str(comm.Get_rank())                        +  
                    "\".")
    
    else:
        l_comm = None
        logger.error("Missing an \"MPI.Intracomm\". Setted "      +
                     ("\"self._comm\" " if type == "local" 
                                        else "\"self._comm_w\" ") +
                     "to None.")

    return l_comm

def check_octree(octree, 
                 comm  ,
                 logger,
                 type = "local"):
    from_para_tree = False
    py_base_name = "Py_Para_Tree"

    for base in octree.__class__.__bases__:
        if (py_base_name in base.__name__):
            from_para_tree = True
            break
        
    if from_para_tree:
        l_octree = octree
        logger.info("Setted octree for "                        +
                    ("local " if type == "local" else "world ") +
                    "comm \""                                   +
                    str(comm.Get_name() if comm else None)      +
                    "\" and rank \""                            +
                    str(comm.Get_rank() if comm else None)      + 
                    "\".")
    
    else:
        l_octree = None
        logger.error("First parameter has not as base class the needed one" +
                     "\"Py_Para_Tree\". \"self._octree\" setted to \"None\".") 

    return l_octree

def is_point_inside_polygons(point   ,
                             polygons,
                             logger  ,
                             log_file,
                             threshold = 0.0):
    inside = False

    if isinstance(polygons, list):
        for i, polygon in enumerate(polygons):
            inside = is_point_inside_polygon(point   ,
                                             polygon ,
                                             logger  ,
                                             log_file,
                                             threshold)
            if (inside):
                return (inside, i)
    else:
        logger = check_null_logger(logger,
                                   log_file)
        logger.error("Second parameter must be a list of lists.")
    return (inside, None)
        

# Determine if a point is inside a given polygon or not.
# https://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html ---> Better link
# http://www.ariel.com.au/a/python-point-int-poly.html
# http://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon
def is_point_inside_polygon(point   ,
                            polygon ,
                            logger  ,
                            log_file,
                            threshold = 0.0):

    n_vert = len(polygon)
    x, y = point
    inside = False

    if isinstance(polygon, list):
        for i in xrange(0, n_vert):
            if (i == 0):
                j = n_vert - 1
            else:
                j = i - 1
            i_x, i_y = polygon[i]
            j_x, j_y = polygon[j]
            if (((i_y > y) != (j_y > y)) and
                ((x + threshold ) < 
                 (((j_x - i_x) * (y - i_y)) / (j_y - i_y)) + i_x)):
                inside = not inside
        return inside
    else:
        logger = check_null_logger(logger, 
                                   log_file)
        logger.error("Second parameter must be a list.")

    return inside

# https://it.wikipedia.org/wiki/Metodo_dei_minimi_quadrati
def least_squares(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] points       ,
                  numpy.ndarray[dtype = numpy.float64_t, ndim = 1] unknown_point,
                  int dim = 2):
    # In 2D we approximate our function as a plane: \"ax + by + c\", in 3D the
    # approximation will be: \"ax + by + cz + d\".
    cdef int n_points = points.shape[0]
    cdef int n_cols = dim + 1
    cdef size_t i
    cdef size_t j

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] A = \
         numpy.zeros(shape = (n_points, n_cols), \
                     dtype = numpy.float64)

    for i in range(n_points):
        for j in range(dim):
            A[i][j] = points[i][j]
        A[i][dim] = 1

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] At = A.T
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] AtA = numpy.dot(At, A)
    # Pseudo-inverse matrix.
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] p = \
         numpy.dot(numpy.linalg.inv(AtA), At)

    # Multiplying \"a\" time \"x\".
    p[0, :] = p[0, :] * unknown_point[0]
    # Multiplying \"b\" time \"y\".
    p[1, :] = p[1, :] * unknown_point[1]
    if (dim == 3):
        # Multiplying \"c\" time \"z\".
        p[2, :] = p[2, :] * unknown_point[2]

    coeffs = numpy.sum(p, axis = 0)

    return coeffs
    
# Perspective transformation coefficients (linear coefficients).
def p_t_coeffs(int dimension                                        ,
               numpy.ndarray[dtype = numpy.float64_t, ndim = 2] o_ps,  # Original points
               numpy.ndarray[dtype = numpy.float64_t, ndim = 2] t_ps,  # Transformed points
               numpy.float64_t dil_z = 1.0):
    # Dimension of the matrix.
    cdef int d_matrix = 8

    cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] matrix = \
         numpy.zeros((d_matrix, d_matrix), dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1 ] rhs = \
         numpy.zeros((d_matrix, ), dtype = numpy.float64)

    matrix[0, :] = [o_ps[0][0], o_ps[0][1], 1, 0, 0, 0, -(o_ps[0][0] * t_ps[0][0]), -(o_ps[0][1] * t_ps[0][0])]
    matrix[1, :] = [o_ps[1][0], o_ps[1][1], 1, 0, 0, 0, -(o_ps[1][0] * t_ps[1][0]), -(o_ps[1][1] * t_ps[1][0])]
    matrix[2, :] = [o_ps[2][0], o_ps[2][1], 1, 0, 0, 0, -(o_ps[2][0] * t_ps[2][0]), -(o_ps[2][1] * t_ps[2][0])]
    matrix[3, :] = [o_ps[3][0], o_ps[3][1], 1, 0, 0, 0, -(o_ps[3][0] * t_ps[3][0]), -(o_ps[3][1] * t_ps[3][0])]
    matrix[4, :] = [0, 0, 0, o_ps[0][0], o_ps[0][1], 1, -(o_ps[0][0] * t_ps[0][1]), -(o_ps[0][1] * t_ps[0][1])]
    matrix[5, :] = [0, 0, 0, o_ps[1][0], o_ps[1][1], 1, -(o_ps[1][0] * t_ps[1][1]), -(o_ps[1][1] * t_ps[1][1])]
    matrix[6, :] = [0, 0, 0, o_ps[2][0], o_ps[2][1], 1, -(o_ps[2][0] * t_ps[2][1]), -(o_ps[2][1] * t_ps[2][1])]
    matrix[7, :] = [0, 0, 0, o_ps[3][0], o_ps[3][1], 1, -(o_ps[3][0] * t_ps[3][1]), -(o_ps[3][1] * t_ps[3][1])]

    rhs[:] = [t_ps[0][0], t_ps[1][0], t_ps[2][0], t_ps[3][0], t_ps[0][1], t_ps[1][1], t_ps[2][1], t_ps[3][1]]

    cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] coefficients = \
         numpy.linalg.solve(matrix, rhs)
    # \"append\" does not occur in place:
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.append.html
    # We append 1 as coefficients \"a33\" (or \"a44\", depending on problem's
    # dimension) without loss of generality.
    coefficients = numpy.append(coefficients, 1)
    cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] r_coefficients = \
         numpy.ndarray(shape = (3, 3)       ,
                       buffer = coefficients,
                       dtype = numpy.float64).T

    if (dimension == 3):
        r_coefficients = numpy.insert(r_coefficients, 2, [0, 0, 0], axis = 1)
        r_coefficients = numpy.insert(r_coefficients, 2, [0, 0, dil_z, 0], axis = 0)

    return r_coefficients

# Perspective transformation coefficients for adjoint matrix.
def p_t_coeffs_adj(int dimension,
                   numpy.ndarray[dtype = numpy.float64_t, ndim = 2] p_t_m): #Perspective transofrmation matrix
    cdef numpy.float64_t det_p_t_m = numpy.linalg.det(p_t_m)
    # Dimension of the adjoint matrix.
    cdef int d_adj_matrix
    cdef numpy.float64_t ad00
    cdef numpy.float64_t ad01
    cdef numpy.float64_t ad02
    cdef numpy.float64_t ad03
    cdef numpy.float64_t ad10
    cdef numpy.float64_t ad11
    cdef numpy.float64_t ad12
    cdef numpy.float64_t ad13
    cdef numpy.float64_t ad20
    cdef numpy.float64_t ad21
    cdef numpy.float64_t ad22
    cdef numpy.float64_t ad23
    cdef numpy.float64_t ad30
    cdef numpy.float64_t ad31
    cdef numpy.float64_t ad32
    cdef numpy.float64_t ad33
    

    if (dimension == 2):
        d_adj_matrix = 3
    else:
        d_adj_matrix = 4

    cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] adj_matrix = \
         numpy.zeros((d_adj_matrix, d_adj_matrix), dtype = numpy.float64)

    if (dimension == 2):
        # Adjoint matrix
             adj_matrix[0, :] = [(p_t_m[1][1] * p_t_m[2][2]) - (p_t_m[1][2] * p_t_m[2][1]),
                                 (p_t_m[0][2] * p_t_m[2][1]) - (p_t_m[0][1] * p_t_m[2][2]),
                                 (p_t_m[0][1] * p_t_m[1][2]) - (p_t_m[0][2] * p_t_m[1][1]),
                                ]
             adj_matrix[1, :] = [(p_t_m[1][2] * p_t_m[2][0]) - (p_t_m[1][0] * p_t_m[2][2]),
                                 (p_t_m[0][0] * p_t_m[2][2]) - (p_t_m[0][2] * p_t_m[2][0]),
                                 (p_t_m[0][2] * p_t_m[1][0]) - (p_t_m[0][0] * p_t_m[1][2]),
                                ]
             adj_matrix[2, :] = [(p_t_m[1][0] * p_t_m[2][1]) - (p_t_m[1][1] * p_t_m[2][0]),
                                 (p_t_m[0][1] * p_t_m[2][0]) - (p_t_m[0][0] * p_t_m[2][1]),
                                 (p_t_m[0][0] * p_t_m[1][1]) - (p_t_m[0][1] * p_t_m[1][0]),
                                ]
    # Dim = 3.
    else:
        ad00 = (p_t_m[1][2] * p_t_m[2][3] * p_t_m[3][1]) - (p_t_m[1][3] * p_t_m[2][2] * p_t_m[3][1]) + (p_t_m[1][3] * p_t_m[2][1] * p_t_m[3][2]) - \
               (p_t_m[1][1] * p_t_m[2][3] * p_t_m[3][2]) - (p_t_m[1][2] * p_t_m[2][1] * p_t_m[3][3]) + (p_t_m[1][1] * p_t_m[2][2] * p_t_m[3][3])
        ad01 = (p_t_m[0][3] * p_t_m[2][2] * p_t_m[3][1]) - (p_t_m[0][2] * p_t_m[2][3] * p_t_m[3][1]) - (p_t_m[0][3] * p_t_m[2][1] * p_t_m[3][2]) + \
               (p_t_m[0][1] * p_t_m[2][3] * p_t_m[3][2]) + (p_t_m[0][2] * p_t_m[2][1] * p_t_m[3][3]) - (p_t_m[0][1] * p_t_m[2][2] * p_t_m[3][3])
        ad02 = (p_t_m[0][2] * p_t_m[1][3] * p_t_m[3][1]) - (p_t_m[0][3] * p_t_m[1][2] * p_t_m[3][1]) + (p_t_m[0][3] * p_t_m[1][1] * p_t_m[3][2]) - \
               (p_t_m[0][1] * p_t_m[1][3] * p_t_m[3][2]) - (p_t_m[0][2] * p_t_m[1][1] * p_t_m[3][3]) + (p_t_m[0][1] * p_t_m[1][2] * p_t_m[3][3])
        ad03 = (p_t_m[0][3] * p_t_m[1][2] * p_t_m[2][1]) - (p_t_m[0][2] * p_t_m[1][3] * p_t_m[2][1]) - (p_t_m[0][3] * p_t_m[1][1] * p_t_m[2][2]) + \
               (p_t_m[0][1] * p_t_m[1][3] * p_t_m[2][2]) + (p_t_m[0][2] * p_t_m[1][1] * p_t_m[2][3]) - (p_t_m[0][1] * p_t_m[1][2] * p_t_m[2][3])
        ad10 = (p_t_m[1][3] * p_t_m[2][2] * p_t_m[3][0]) - (p_t_m[1][2] * p_t_m[2][3] * p_t_m[3][0]) - (p_t_m[1][3] * p_t_m[2][0] * p_t_m[3][2]) + \
               (p_t_m[1][0] * p_t_m[2][3] * p_t_m[3][2]) + (p_t_m[1][2] * p_t_m[2][0] * p_t_m[3][3]) - (p_t_m[1][0] * p_t_m[2][2] * p_t_m[3][3])
        ad11 = (p_t_m[0][2] * p_t_m[2][3] * p_t_m[3][0]) - (p_t_m[0][3] * p_t_m[2][2] * p_t_m[3][0]) + (p_t_m[0][3] * p_t_m[2][0] * p_t_m[3][2]) - \
               (p_t_m[0][0] * p_t_m[2][3] * p_t_m[3][2]) - (p_t_m[0][2] * p_t_m[2][0] * p_t_m[3][3]) + (p_t_m[0][0] * p_t_m[2][2] * p_t_m[3][3])
        ad12 = (p_t_m[0][3] * p_t_m[1][2] * p_t_m[3][0]) - (p_t_m[0][2] * p_t_m[1][3] * p_t_m[3][0]) - (p_t_m[0][3] * p_t_m[1][0] * p_t_m[3][2]) + \
               (p_t_m[0][0] * p_t_m[1][3] * p_t_m[3][2]) + (p_t_m[0][2] * p_t_m[1][0] * p_t_m[3][3]) - (p_t_m[0][0] * p_t_m[1][2] * p_t_m[3][3])
        ad13 = (p_t_m[0][2] * p_t_m[1][3] * p_t_m[2][0]) - (p_t_m[0][3] * p_t_m[1][2] * p_t_m[2][0]) + (p_t_m[0][3] * p_t_m[1][0] * p_t_m[2][2]) - \
               (p_t_m[0][0] * p_t_m[1][3] * p_t_m[2][2]) - (p_t_m[0][2] * p_t_m[1][0] * p_t_m[2][3]) + (p_t_m[0][0] * p_t_m[1][2] * p_t_m[2][3])
        ad20 = (p_t_m[1][1] * p_t_m[2][3] * p_t_m[3][0]) - (p_t_m[1][3] * p_t_m[2][1] * p_t_m[3][0]) + (p_t_m[1][3] * p_t_m[2][0] * p_t_m[3][1]) - \
               (p_t_m[1][0] * p_t_m[2][3] * p_t_m[3][1]) - (p_t_m[1][1] * p_t_m[2][0] * p_t_m[3][3]) + (p_t_m[1][0] * p_t_m[2][1] * p_t_m[3][3])
        ad21 = (p_t_m[0][3] * p_t_m[2][1] * p_t_m[3][0]) - (p_t_m[0][1] * p_t_m[2][3] * p_t_m[3][0]) - (p_t_m[0][3] * p_t_m[2][0] * p_t_m[3][1]) + \
               (p_t_m[0][0] * p_t_m[2][3] * p_t_m[3][1]) + (p_t_m[0][1] * p_t_m[2][0] * p_t_m[3][3]) - (p_t_m[0][0] * p_t_m[2][1] * p_t_m[3][3])
        ad22 = (p_t_m[0][1] * p_t_m[1][3] * p_t_m[3][0]) - (p_t_m[0][3] * p_t_m[1][1] * p_t_m[3][0]) + (p_t_m[0][3] * p_t_m[1][0] * p_t_m[3][1]) - \
               (p_t_m[0][0] * p_t_m[1][3] * p_t_m[3][1]) - (p_t_m[0][1] * p_t_m[1][0] * p_t_m[3][3]) + (p_t_m[0][0] * p_t_m[1][1] * p_t_m[3][3])
        ad23 = (p_t_m[0][3] * p_t_m[1][1] * p_t_m[2][0]) - (p_t_m[0][1] * p_t_m[1][3] * p_t_m[2][0]) - (p_t_m[0][3] * p_t_m[1][0] * p_t_m[2][1]) + \
               (p_t_m[0][0] * p_t_m[1][3] * p_t_m[2][1]) + (p_t_m[0][1] * p_t_m[1][0] * p_t_m[2][3]) - (p_t_m[0][0] * p_t_m[1][1] * p_t_m[2][3])
        ad30 = (p_t_m[1][2] * p_t_m[2][1] * p_t_m[3][0]) - (p_t_m[1][1] * p_t_m[2][2] * p_t_m[3][0]) - (p_t_m[1][2] * p_t_m[2][0] * p_t_m[3][1]) + \
               (p_t_m[1][0] * p_t_m[2][2] * p_t_m[3][1]) + (p_t_m[1][1] * p_t_m[2][0] * p_t_m[3][2]) - (p_t_m[1][0] * p_t_m[2][1] * p_t_m[3][2])
        ad31 = (p_t_m[0][1] * p_t_m[2][2] * p_t_m[3][0]) - (p_t_m[0][2] * p_t_m[2][1] * p_t_m[3][0]) + (p_t_m[0][2] * p_t_m[2][0] * p_t_m[3][1]) - \
               (p_t_m[0][0] * p_t_m[2][2] * p_t_m[3][1]) - (p_t_m[0][1] * p_t_m[2][0] * p_t_m[3][2]) + (p_t_m[0][0] * p_t_m[2][1] * p_t_m[3][2])
        ad32 = (p_t_m[0][2] * p_t_m[1][1] * p_t_m[3][0]) - (p_t_m[0][1] * p_t_m[1][2] * p_t_m[3][0]) - (p_t_m[0][2] * p_t_m[1][0] * p_t_m[3][1]) + \
               (p_t_m[0][0] * p_t_m[1][2] * p_t_m[3][1]) + (p_t_m[0][1] * p_t_m[1][0] * p_t_m[3][2]) - (p_t_m[0][0] * p_t_m[1][1] * p_t_m[3][2])
        ad33 = (p_t_m[0][1] * p_t_m[1][2] * p_t_m[2][0]) - (p_t_m[0][2] * p_t_m[1][1] * p_t_m[2][0]) + (p_t_m[0][2] * p_t_m[1][0] * p_t_m[2][1]) - \
               (p_t_m[0][0] * p_t_m[1][2] * p_t_m[2][1]) - (p_t_m[0][1] * p_t_m[1][0] * p_t_m[2][2]) + (p_t_m[0][0] * p_t_m[1][1] * p_t_m[2][2])

        adj_matrix[0, :] = [ad00, ad01, ad02, ad03]
        adj_matrix[1, :] = [ad10, ad11, ad12, ad13]
        adj_matrix[2, :] = [ad20, ad21, ad22, ad23]
        adj_matrix[3, :] = [ad30, ad31, ad32, ad33]

    adj_matrix = numpy.true_divide(adj_matrix, det_p_t_m)

    return adj_matrix

def metric_coefficients(dimension          ,
                        in_points          ,
                        matrix_coefficients,
                        logger             ,
                        log_file):
    logger = check_null_logger(logger, log_file)
    dim = dimension
    A = matrix_coefficients
    # http://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    if (type(in_points).__module__ == numpy.__name__):
        points = in_points
    else: 
        points = numpy.array(in_points)
    # Denominators.
    # A02*x + A12*y...
    dens = (numpy.add(numpy.multiply(points[:, 0], A[0][dim]), \
                      numpy.multiply(points[:, 1], A[1][dim])))
    # ...+ A22.
    dens = numpy.add(dens, A[dim][dim])
    # Part 01 of numerators for epsilon coordinate.
    nums_epsilon_01 = dens 
    # Part 01 of numerators for nu coordinate.
    nums_nu_01 = dens 
    # (A02*x + A12*y + A22)^2.
    dens2 = numpy.square(dens)
    # (A02*x + A12*y + A22)^4.
    dens4 = numpy.square(dens2)
    # Part 02 of numerators for epsilon coordinate.
    # A00*x + A10*y...
    nums_epsilon_02 = (numpy.add(numpy.multiply(points[:, 0], A[0][0]), \
                                 numpy.multiply(points[:, 1], A[1][0])))
    # ...+ A20
    nums_epsilon_02 = numpy.add(nums_epsilon_02, A[dim][0])
    # Part 02 of numerators for nu coordinate.
    # A01*x + A11*y...
    nums_nu_02 = (numpy.add(numpy.multiply(points[:, 0], A[0][1]), \
                            numpy.multiply(points[:, 1], A[1][1])))
    # ...+ A21
    nums_nu_02 = numpy.add(nums_nu_02, A[dim][1])

    ds_epsilon_x = numpy.true_divide(numpy.subtract(numpy.multiply(nums_epsilon_01, A[0][0]),
                                                    numpy.multiply(nums_epsilon_02, A[0][dim])),
                                     dens2)
    ds_epsilon_y = numpy.true_divide(numpy.subtract(numpy.multiply(nums_epsilon_01, A[1][0]),
                                                    numpy.multiply(nums_epsilon_02, A[1][dim])),
                                     dens2)
    ds_nu_x = numpy.true_divide(numpy.subtract(numpy.multiply(nums_nu_01, A[0][1]),
                                               numpy.multiply(nums_nu_02, A[0][dim])),
                                dens2)
    ds_nu_y = numpy.true_divide(numpy.subtract(numpy.multiply(nums_nu_01, A[1][1]),
                                               numpy.multiply(nums_nu_02, A[1][dim])),
                                dens2)
    ds2_epsilon_x = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[0][2],
                                                                    dens),
                                                     numpy.sum([numpy.multiply(A[0][0]*A[1][2] - A[0][2]*A[1][0],
                                                                              points[:, 1]),
                                                               (A[0][0]*A[2][2] - A[0][2]*A[2][0])])),
                                      dens4)
    ds2_epsilon_y = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[1][2],
                                                                    dens),
                                                     numpy.sum([numpy.multiply(A[1][0]*A[0][2] - A[1][2]*A[0][0],
                                                                              points[:, 0]),
                                                               (A[1][0]*A[2][2] - A[1][2]*A[2][0])])),
                                      dens4)
    ds2_nu_x = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[0][2],
                                                               dens),
                                                numpy.sum([numpy.multiply(A[0][1]*A[1][2] - A[0][2]*A[1][1],
                                                                         points[:, 1]),
                                                          (A[0][1]*A[2][2] - A[0][2]*A[2][1])])),
                                 dens4)
    ds2_nu_y = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[1][2],
                                                               dens),
                                                numpy.sum([numpy.multiply(A[1][1]*A[0][2] - A[1][2]*A[0][1],
                                                                         points[:, 0]),
                                                          (A[1][1]*A[2][2] - A[1][2]*A[2][1])])),
                                      dens4)
                                                                    
    # Metric coefficients.
    m_cs = [ds_epsilon_x, ds_epsilon_y, ds_nu_x, ds_nu_y, ds2_epsilon_x, ds2_epsilon_y, ds2_nu_x, ds2_nu_y]
    # Numpy metric coefficients.
    n_m_cs = numpy.array(m_cs)

    return n_m_cs
       
def apply_persp_trans_inv(int dimension                                                ,
                          numpy.ndarray[dtype = numpy.float64_t, ndim = 1] point       ,
                          numpy.ndarray[dtype = numpy.float64_t, ndim = 2] coefficients,
                          bool r_a_n_d = False):
    # Numpy point.
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] np_point = \
         numpy.zeros(shape = (dimension + 1,), \
                     dtype = numpy.float64)
    cdef size_t i
    cdef float divisor = 0.0
    cdef float w_first

    for i in range(dimension):
        divisor = divisor + coefficients[i][dimension] * point[i]
    divisor = divisor + coefficients[dimension][dimension]
    w_first = 1.0 / divisor

    for i in range(dimension):
        np_point[i] = point[i] * w_first
    # Homogeneous coordinates.
    np_point[dimension] = 1.0 * w_first

    # Numpy transformed inverse point.
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] np_t_i_point = \
         numpy.dot(np_point, coefficients)

    # Transformed inverse point.
    t_i_point = [0.0] * dimension
    for i in range(dimension):
        t_i_point[i] = np_t_i_point[i]

    if (r_a_n_d):
        return t_i_point, np_t_i_point[0 : dimension]

    return t_i_point

def apply_persp_trans(int dimension                                                ,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 1] point       ,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 2] coefficients,
                      bool r_a_n_d = False):
    # Numpy point.
    # http://stackoverflow.com/questions/14415741/numpy-array-vs-asarray
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] np_point = \
         numpy.zeros(shape = (dimension + 1,), \
                     dtype = numpy.float64)
    cdef size_t i

    for i in range(dimension):
        np_point[i] = point[i]
    # Homogeneous coordinates.
    np_point[dimension] = 1

    # Numpy transformed point.
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] np_t_point = \
         numpy.dot(np_point, coefficients)
    np_t_point = numpy.true_divide(np_t_point,
                                   np_t_point[dimension])

    # Transformed point.
    t_point = [0.0] * dimension
    for i in range(dimension):
        t_point[i] = np_t_point[i]

    if (r_a_n_d):
        return t_point, np_t_point[0 : dimension]

    return t_point

def join_strings(*args):
    # List of strings to join.
    strs_list = []
    map(strs_list.append, args)
    #strs_list = [str(arg) for arg in args]
    # Returned string.
    r_s = "".join(strs_list)
    return r_s
# ------------------------------------------------------------------------------

# ------------------------------------LOGGER------------------------------------
class Logger(object):
    def __init__(self    , 
                 name    , 
                 log_file,
                 what_log):
        self._logger = logging.getLogger(name)
        # http://stackoverflow.com/questions/15870380/python-custom-logging-across-all-modules
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            # http://stackoverflow.com/questions/2266646/how-to-i-disable-and-re-enable-console-logging-in-python
            if (what_log == "critical"):
                self._logger.setLevel(logging.CRITICAL)
            self._handler = logging.FileHandler(log_file)

            self._formatter = logging.Formatter("%(name)15s - "    + 
                                                "%(asctime)s - "   +
                                                "%(funcName)8s - " +
                                                "%(levelname)s - " +
                                                "%(message)s")
            self._handler.setFormatter(self._formatter)
            self._logger.addHandler(self._handler)
            self._logger.propagate = False

    @property
    def logger(self):
        return self._logger
# ------------------------------------------------------------------------------
