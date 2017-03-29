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
        # \"Python\"\"trim\" method is called \"strip\":
        # http://stackoverflow.com/questions/1185524/how-to-trim-whitespace-including-tabs
        return [int(number.strip(' \t\n\r')) if integer else float(number) \
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
        return [get_list_from_string(string_chunk.strip(' \t\n\r'),
                                     splitter_for_list            ,
                                     integer)        \
                for string_chunk in string.split(splitter_for_lists)]
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

def is_point_inside_polygons(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] point   ,
                             numpy.ndarray[dtype = numpy.float64_t, ndim = 4] polygons,
                             int dimension = 2):
    cdef bool inside = False
    cdef size_t i
    cdef int n_polys = polygons.shape[0]
    cdef int no_poly = -1

    for i in xrange(n_polys):
        inside = is_point_inside_polygon(point      ,
                                         polygons[i],
                                         dimension)
        if (inside):
            return (inside, i)

    return (inside, no_poly)

# Determine if a point is inside a given polygon or not.
# https://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html ---> Better link
# http://www.ariel.com.au/a/python-point-int-poly.html
# http://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon
def is_point_inside_polygon(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] point  ,
                            numpy.ndarray[dtype = numpy.float64_t, ndim = 3] polygon,
                            int dimension = 2):

    cdef int n_verts_face = 4
    cdef int n_faces = 1 if (dimension == 2) else 6
    cdef size_t i
    cdef size_t j
    cdef size_t face
    cdef size_t x, y
    cdef double p_x, p_y
    cdef bool inside = False
    cdef double i_x, i_y, j_x, j_y

    #faces = [polygon] if (dimension == 2) else \
    #        [[polygon[0], polygon[1], polygon[2], polygon[3]],
    #         [polygon[4], polygon[5], polygon[6], polygon[7]],
    #         [polygon[0], polygon[4], polygon[2], polygon[6]],
    #         [polygon[1], polygon[5], polygon[3], polygon[7]],
    #         [polygon[0], polygon[1], polygon[4], polygon[5]],
    #         [polygon[2], polygon[3], polygon[6], polygon[7]]]

    #cdef numpy.ndarray[dtype = numpy.float64_t,
    #                   ndim = 3] faces = \
    #     numpy.ndarray(shape = (n_faces, n_verts_face, dimension),            \
    #                   buffer = numpy.array(polygon) if (dimension == 2) else \
    #                            numpy.array(polygon[0], polygon[1],
    #                                        polygon[2], polygon[3],
    #                                        polygon[4], polygon[5],
    #                                        polygon[6], polygon[7],
    #                                        polygon[0], polygon[4],
    #                                        polygon[2], polygon[6],
    #                                        polygon[1], polygon[5],
    #                                        polygon[3], polygon[7],
    #                                        polygon[0], polygon[1],
    #                                        polygon[4], polygon[5],
    #                                        polygon[2], polygon[3],
    #                                        polygon[6], polygon[7]),
    #                   dtype = numpy.float64)

    for face in xrange(0, n_faces):
        if (face < 2):
            x = 0
            y = 1
        elif (1 < face < 4):
            x = 2
        else:
            x = 0
            y = 2

        p_x = point[0][x]
        p_y = point[0][y]

        for i in xrange(0, n_verts_face):
            j = i - 1
            if (i == 0):
                j = n_verts_face - 1

            i_x = polygon[face][i][x]
            i_y = polygon[face][i][y]
            j_x = polygon[face][j][x]
            j_y = polygon[face][j][y]

            if (((i_y > p_y) != (j_y > p_y)) and
                (p_x <
                 (((j_x - i_x) * (p_y - i_y)) / (j_y - i_y)) + i_x)):
                inside = not inside
        if (not inside):
            break

    return inside

# http://www.ripmat.it/mate/d/dc/dcee.html
# http://stackoverflow.com/questions/11907947/how-to-check-if-a-point-lies-on-a-line-between-2-other-points
def is_point_on_line(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] point      ,
                     numpy.ndarray[dtype = numpy.float64_t, ndim = 2] line_points,
                     int dimension = 2):

    absolute = numpy.absolute

    cdef size_t i
    cdef n_points = 2
    cdef double d_x
    cdef double d_y
    cdef double d_yc = point[0][1] - line_points[0][1]
    cdef double d_xc = point[0][0] - line_points[0][0]
    cdef double d_x1 = line_points[1][0] - line_points[0][0]
    cdef double d_y1 = line_points[1][1] - line_points[0][1]
    cdef double threshold = 1.0e-15
    cdef double diff
    cdef bool on_line = False

    for i in xrange(0, n_points):
        d_x = absolute(point[0][0] - line_points[i][0])
        d_y = absolute(point[0][1] - line_points[i][1])

        if ((d_x <= threshold) and (d_y <= threshold)):
            on_line = True

            return on_line

    diff = absolute((d_yc * d_x1) - (d_y1 * d_xc))

    on_line = (diff <= threshold)

    return on_line

def is_point_on_lines(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] point      ,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 3] line_points,
                      int dimension = 2):

    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef int n_lines = 4 if (dimension == 2) else 12
    cdef int n_polys = line_points.shape[0]
    cdef bool on_lines = False
    # TODO: Modify for 3D case.
    for k in xrange(0, n_polys):
        for i in xrange(0, n_lines):
            j = (i + 1) if (i < 3) else 0
            on_lines = is_point_on_line(point                           ,
                                        numpy.array((line_points[k][i],
                                                     line_points[k][j])),
                                        dimension)
            if (on_lines):
                return on_lines

    return on_lines
# TODO: extend to 3D.
def exact_sol(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] l_points ,
              numpy.ndarray[dtype = numpy.float64_t, ndim = 1] alpha    ,
              numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta     ,
              int dim = 2):
    cdef int n_points = l_points.shape[0]
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] sol =          \
         numpy.zeros(n_points                 , \
                     dtype = numpy.float64)
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] p_points =     \
         numpy.zeros((n_points, dim)          , \
                     dtype = numpy.float64)

    apply_bil_mapping(l_points,
                      alpha   ,
                      beta    ,
                      p_points,
                      dim)
    nsin = numpy.sin
    npower = numpy.power
    nadd = numpy.add
    # sin((x - 0.5)^2 + (y - 0.5)^2).
    numpy.copyto(sol,
                 nsin(nadd(npower(nadd(p_points[:, 0], -0.5), 2),
                           npower(nadd(p_points[:, 1], -0.5), 2))))

    return sol

# TODO: extend to 3D.
def exact_2nd_der(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] l_points ,
                  numpy.ndarray[dtype = numpy.float64_t, ndim = 1] alpha    ,
                  numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta     ,
                  int dim = 2):
    cdef int n_points = l_points.shape[0]
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] p_points =     \
         numpy.zeros((n_points, dim)          , \
                     dtype = numpy.float64)

    apply_bil_mapping(l_points,
                      alpha   ,
                      beta    ,
                      p_points,
                      dim)
    nsin = numpy.sin
    ncos = numpy.cos
    npower = numpy.power
    nadd = numpy.add
    nmul = numpy.multiply
    # 4 * cos((x - 0.5)^2 + (y - 0.5)^2) -
    # 4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
    # ((x - 0.5)^2 + (y - 0.5)^2).
    return nadd(nmul(4.0,
                     ncos(nadd(npower(nadd(p_points[:, 0], -0.5),
                                      2),
                               npower(nadd(p_points[:, 1], -0.5),
                                      2)))),
                nmul(nmul(-4.0,
                          nsin(nadd(npower(nadd(p_points[:, 0], -0.5),
                                           2),
                                    npower(nadd(p_points[:, 1], -0.5),
                                           2)))),
                     nadd(npower(nadd(p_points[:, 0], -0.5),
                                 2),
                          npower(nadd(p_points[:, 1], -0.5),
                                 2))))

def check_oct_corners(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] numpy_corners,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 1] alpha        ,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta         ,
                      numpy.ndarray[dtype = numpy.float64_t, ndim = 4] polygons     ,
                      int dim = 2):
    cdef bool penalized = True
    cdef bool is_corner_penalized
    cdef int n_oct_corners = 4 if (dim == 2) else 8
    cdef int n_polygon
    cdef size_t i
    # Getting a \"numpy\" array of shape (3, 1): \"array([[0.], [0.], [0.]])\"
    # (each element of the array is a \"numpy\" array of \"ndim\" = 1 and
    # \"shape\" = (1, )).
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] n_t_corner =   \
         numpy.zeros(shape = (3, 1),            \
                     dtype = numpy.float64)
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] numpy_corner = \
         numpy.zeros(shape = (1, 3),            \
                     dtype = numpy.float64)

    for i in xrange(n_oct_corners):
        # Getting a \"numpy\" array of \"ndim\" = 2.
        numpy.copyto(numpy_corner[0],
                     numpy_corners[i])
        apply_bil_mapping(numpy_corner ,
                          alpha        ,
                          beta         ,
                          n_t_corner   ,
                          dim)
        (is_corner_penalized,
         n_polygon) = is_point_inside_polygons(n_t_corner,
                                               polygons  ,
                                               dim)
        if (not is_corner_penalized):
            penalized = False
            break

    return (penalized, n_polygon)

#https://www.particleincell.com/2012/quad-interpolation/
def bil_mapping(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] nodes,
                numpy.ndarray[dtype = numpy.float64_t, ndim = 1] alpha,
                numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta ,
                bool for_pablo = False                                ,
                int dim = 2):
    # Number of points; in 2D is equal to 4.
    cdef int n_nodes = nodes.shape[0]

    cdef numpy.ndarray[dtype = numpy.float64_t,  \
                       ndim = 2] A =             \
         numpy.zeros(shape = (n_nodes, n_nodes), \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t,\
                       ndim = 1] b_x =         \
         numpy.zeros(shape = (n_nodes),        \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t,\
                       ndim = 1] b_y =         \
         numpy.zeros(shape = (n_nodes),        \
                     dtype = numpy.float64)

    A[0][0] = 1.0
    A[0][1] = 0.0
    A[0][2] = 0.0
    A[0][3] = 0.0
    A[1][0] = 1.0
    A[1][1] = 1.0
    A[1][2] = 0.0
    A[1][3] = 0.0
    A[2][0] = 1.0
    A[2][1] = 1.0
    A[2][2] = 1.0
    A[2][3] = 1.0
    A[3][0] = 1.0
    A[3][1] = 0.0
    A[3][2] = 1.0
    A[3][3] = 0.0
    # Here \"b_x[2]\" and \"b_x[1]\" get inverted values from \"nodes\" because
    # \"PABLO\" is giving us first the neighbours of faces 0 and 1 (perpendi-
    # cular to \"x\") and then of faces 2 and 3 (perpendicular to \"y\").
    # The same things happens for \"b_y\". But, if \"for_pablo\" is \"False\",
    # then no need to apply the inversion.
    b_x[0] = nodes[0][0]
    b_x[1] = nodes[1][0]
    b_x[2] = nodes[2][0]
    b_x[3] = nodes[3][0]

    b_y[0] = nodes[0][1]
    b_y[1] = nodes[1][1]
    b_y[2] = nodes[2][1]
    b_y[3] = nodes[3][1]

    if (for_pablo):
        b_x[1] = nodes[2][0]
        b_x[2] = nodes[1][0]
        b_y[1] = nodes[2][1]
        b_y[2] = nodes[1][1]
    # Finding coefficients for the bilinear mapping function between physical
    # and logical domain: x = alpha_0 + alpha_1*l + alpha_2*m + alpha_3*l*m
    #                     y = beta_0 + beta_1*l + beta_2*m + beta_3*l*m
    # where \"l\" and \"m\" are the coordinates of a logical square of coordi-
    # nates (0, 0), (1, 0), (1, 1), (0, 1) where we will apply the bilinear in-
    # terpolation.
    numpy.copyto(alpha, numpy.linalg.solve(A, b_x))
    numpy.copyto(beta, numpy.linalg.solve(A, b_y))

def apply_bil_mapping_inv(numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 2] p_points     , # physical points
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] alpha        ,
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] beta         ,
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 2] l_points     , # logical points
                          int dim = 2):
    # Inverting the following equation (for the \"ys\" is the same but with
    # betas):
    # x = alpha_0 + alpha_1*l + alpha_2*m + alpha_3*l*m (2D)
    # x = alpha_0 + alpha_1*l + alpha_2*m + alpha_3*n + alpha_4*l*m +
    #     alpha_5*l*n + alpha_6*m*n (3D)
    cdef double a_m
    cdef size_t i
    cdef int n_points = p_points.shape[0]
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] b_m =          \
         numpy.zeros(shape = (n_points),        \
                     dtype = numpy.float64)
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] c_m =          \
         numpy.zeros(shape = (n_points),        \
                     dtype = numpy.float64)
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] m_1 =          \
         numpy.zeros(shape = (n_points),        \
                     dtype = numpy.float64)
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] m_2 =          \
         numpy.zeros(shape = (n_points),        \
                     dtype = numpy.float64)
    nadd = numpy.add
    nmul = numpy.multiply
    ntdivide = numpy.true_divide
    # Defining inverse mapping from physical to logical.
    a_m = nadd(nmul(alpha[3], beta[2]),
               nmul(-1.0, nmul(alpha[2], beta[3])))
    numpy.copyto(b_m,
                 nadd(nmul(alpha[3], beta[0]),
                      nadd(nmul(-1.0, nmul(alpha[0], beta[3])),
                           nadd(nmul(alpha[1], beta[2]),
                                nadd(nmul(-1.0, nmul(alpha[2], beta[1])),
                                     nadd(nmul(p_points[:, 0], beta[3]),
                                          nmul(-1.0, nmul(p_points[:, 1], alpha[3]))))))))
    numpy.copyto(c_m,
                 nadd(nmul(alpha[1], beta[0]),
                      nadd(nmul(-1.0, nmul(alpha[0], beta[1])),
                           nadd(nmul(p_points[:, 0], beta[1]),
                                nmul(-1.0, nmul(p_points[:, 1], alpha[1]))))))
    # Returning logical coordinates.
    if (not a_m):
        numpy.copyto(l_points[:, 1], ntdivide(nmul(-1.0, c_m), b_m))
    else:
        # \"m\" is solution of a second order equation, so we have two solu-
        # tions...
        numpy.copyto(m_1,
                     ntdivide(nadd(nmul(-1.0, b_m),
                                   numpy.sqrt(nadd(nmul(b_m, b_m),
                                                   nmul(-4.0, nmul(a_m, c_m))))),
                              nmul(2.0, a_m)))
        numpy.copyto(m_2,
                     ntdivide(nadd(nmul(-1.0, b_m),
                                   nmul(-1.0, numpy.sqrt(nadd(nmul(b_m, b_m),
                                                              nmul(-4.0, nmul(a_m, c_m)))))),
                              nmul(2.0, a_m)))
        # ...but we choose the right value for \"m\", respecting the logical do-
        # main.
        for i in xrange(0, n_points):
            if (0.0 <= m_1[i] <= 1.0):
                l_points[i, 1] = m_1[i]
            else:
                l_points[i, 1] = m_2[i]
    numpy.copyto(l_points[:, 0],
                 ntdivide(nadd(p_points[:, 0],
                               nadd(nmul(-1.0, alpha[0]),
                                    nmul(-1.0, nmul(alpha[2], l_points[:, 1])))),
                          nadd(alpha[1],
                               nmul(alpha[3], l_points[:, 1]))))

def get_points_local_ring(numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] point        ,
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] oct_center   ,
                          int dim = 2):
    # Index of quadrant.
    cdef int ind_quad
    cdef double x_p = point[0]
    cdef double y_p = point[1]
    cdef double x_c = oct_center[0]
    cdef double y_c = oct_center[1]
    cdef double d_x = (x_p - x_c)
    cdef double d_y = (y_p - y_c)
    cdef numpy.ndarray[dtype = numpy.uint8_t, mode = "c", ndim = 1] l_ring = \
         numpy.zeros((3, ), dtype = numpy.uint8)

    if ((d_x >= 0.0) and (d_y >= 0.0)):
        ind_quad = 0
        l_ring[0] = 1 # Face
        l_ring[1] = 3 # Node
        l_ring[2] = 3 # Face
    elif ((d_x < 0.0) and (d_y >= 0.0)):
        ind_quad = 1
        l_ring[0] = 0 # Face
        l_ring[1] = 2 # Node
        l_ring[2] = 3 # Face
    elif ((d_x <= 0.0) and (d_y < 0.0)):
        ind_quad = 2
        l_ring[0] = 0 # Face
        l_ring[1] = 0 # Node
        l_ring[2] = 2 # Face
    else:
        ind_quad = 3
        l_ring[0] = 1 # Face
        l_ring[1] = 1 # Node
        l_ring[2] = 2 # Face

    return l_ring.tolist()


def apply_bil_mapping(numpy.ndarray[dtype = numpy.float64_t, \
                                    ndim = 2] l_points     , # logical points
                      numpy.ndarray[dtype = numpy.float64_t, \
                                    ndim = 1] alpha        ,
                      numpy.ndarray[dtype = numpy.float64_t, \
                                    ndim = 1] beta         ,
                      numpy.ndarray[dtype = numpy.float64_t, \
                                    ndim = 2] p_points     , # physical points
                      int dim = 2):
    # Applying the following equation (for the \"ys\" is the same but with
    # betas):
    # x = alpha_0 + alpha_1*l + alpha_2*m + alpha_3*l*m (2D)
    # x = alpha_0 + alpha_1*l + alpha_2*m + alpha_3*n + alpha_4*l*m +
    #     alpha_5*l*n + alpha_6*m*n (3D)
    nadd = numpy.add
    nmul = numpy.multiply
    # Returning physical coordinates.
    numpy.copyto(p_points[:, 0],
                 nadd(nadd(alpha[0], nmul(alpha[1], l_points[:, 0])),
                      nadd(nmul(alpha[2], l_points[:, 1]),
                           nmul(alpha[3], nmul(l_points[:, 0], l_points[:, 1])))))
    numpy.copyto(p_points[:, 1],
                 nadd(nadd(beta[0], nmul(beta[1], l_points[:, 0])),
                      nadd(nmul(beta[2], l_points[:, 1]),
                           nmul(beta[3], nmul(l_points[:, 0], l_points[:, 1])))))

#https://en.wikipedia.org/wiki/Bilinear_interpolation
# We want to obtain the alternative algorithm:
# f(x, y) = b_0*f(x_0, y_0) + b_1*f(x_1, y_1) + b_2*f(x_2, y_2) + b_3*f(x_3, y_3)
# but on the logical square (0, 0), (1, 0), (1, 1), (0, 1)
def bil_coeffs(numpy.ndarray[dtype = numpy.float64_t, ndim = 2] nodes,
               numpy.ndarray[dtype = numpy.float64_t, ndim = 1] point,
               int dim = 2):
    # Number of nodes; in 2D is equal to 4.
    cdef int n_nodes = nodes.shape[0]
    cdef double multiplier
    cdef double l
    cdef double m

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] coeffs =       \
         numpy.zeros(shape = (n_nodes),         \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t,  \
                       ndim = 2] A =             \
         numpy.zeros(shape = (n_nodes, n_nodes), \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] b =            \
         numpy.zeros(shape = (n_nodes),         \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] alpha =        \
         numpy.zeros(shape = (n_nodes),         \
                     dtype = numpy.float64)

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 1] beta =         \
         numpy.zeros(shape = (n_nodes),         \
                     dtype = numpy.float64)
    # A \"numpy\" empty array (size == 0) of shape (0,).
    cdef numpy.ndarray[dtype = numpy.float64_t,
                       ndim = 1] n_e_array = \
         numpy.array([], \
                     dtype = numpy.float64)

    if (nodes.size == 0):
        return n_e_array

    A[0][0] = 1.0
    A[0][1] = nodes[0][0]
    A[0][2] = nodes[0][1]
    A[1][0] = 1.0
    A[1][1] = nodes[1][0]
    A[1][2] = nodes[1][1]
    A[2][0] = 1.0
    A[2][1] = nodes[2][0]
    A[2][2] = nodes[2][1]
    b[0] = 1.0
    b[1] = point[0]
    b[2] = point[1]
    if (n_nodes == 4):
        A[0][3] = nodes[0][0] * nodes[0][1]
        A[1][3] = nodes[1][0] * nodes[1][1]
        A[2][3] = nodes[2][0] * nodes[2][1]
        A[3][0] = 1.0
        A[3][1] = nodes[3][0]
        A[3][2] = nodes[3][1]
        A[3][3] = nodes[3][0] * nodes[3][1]
        b[3] = point[0] * point[1]
    #A[0][0] = 1.0
    #A[0][1] = 0.0
    #A[0][2] = 0.0
    #A[0][3] = 0.0
    #A[1][0] = 1.0
    #A[1][1] = 1.0
    #A[1][2] = 0.0
    #A[1][3] = 0.0
    #A[2][0] = 1.0
    #A[2][1] = 1.0
    #A[2][2] = 1.0
    #A[2][3] = 1.0
    #A[3][0] = 1.0
    #A[3][1] = 0.0
    #A[3][2] = 1.0
    #A[3][3] = 0.0

    #alpha, \
    #beta = bil_mapping(nodes            ,
    #                   for_pablo = False,
    #                   dim = 2)
    ## Getting logical coordinates.
    #l, m = apply_bil_mapping_inv(point,
    #                             alpha,
    #                             beta ,
    #                             dim = 2)
    #b[0] = 1.0
    #b[1] = l
    #b[2] = m
    #b[3] = l * m

    coeffs = numpy.linalg.solve(A.T, b)

    #coeffs[0] = ((nodes[3][0] - point[0]) *
    #             (nodes[3][1] - point[1]))

    #coeffs[1] = ((point[0] - nodes[0][0]) *
    #             (nodes[3][1] - point[1]))

    #coeffs[2] = ((nodes[3][0] - point[0]) *
    #             (point[1] - nodes[0][1]))

    #coeffs[3] = ((point[0] - nodes[0][0]) *
    #             (point[1] - nodes[0][1]))

    #multiplier = 1 / ((nodes[3][0] - nodes[0][0]) *
    #                  (nodes[3][1] - nodes[0][1]))

    #coeffs = multiplier * coeffs

    return coeffs

def jacobians_bil_mapping(numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 2] l_points      , # logical points
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] alpha        ,
                          numpy.ndarray[dtype = numpy.float64_t, \
                                        ndim = 1] beta         ,
                         int dim = 2):
    # Jacobian size.
    cdef int j_size = 2 if (dim == 2) else 3
    cdef size_t n_points = l_points.shape[0]
    cdef size_t i

    cdef numpy.ndarray[dtype = numpy.float64_t,          \
                       ndim = 3] Js =                    \
         numpy.zeros(shape = (n_points, j_size, j_size), \
                     dtype = numpy.float64)

    for i in xrange(0, n_points):
        numpy.copyto(Js[i],
                     jacobian_bil_mapping(l_points[i],
                                          alpha      ,
                                          beta       ,
                                          dim))
    return Js

def jacobian_bil_mapping(numpy.ndarray[dtype = numpy.float64_t, \
                                       ndim = 1] l_point      , # logical point
                         numpy.ndarray[dtype = numpy.float64_t, \
                                       ndim = 1] alpha        ,
                         numpy.ndarray[dtype = numpy.float64_t, \
                                       ndim = 1] beta         ,
                         int dim = 2):
    # Jacobian size.
    cdef int j_size = 2 if (dim == 2) else 3

    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] J =            \
         numpy.zeros(shape = (j_size, j_size),  \
                     dtype = numpy.float64)

    cdef double d_x_d_l
    cdef double d_x_d_m
    cdef double d_y_d_l
    cdef double d_y_d_m

    d_x_d_l = alpha[1] + (alpha[3] * l_point[1])
    d_x_d_m = alpha[2] + (alpha[3] * l_point[0])
    d_y_d_l = beta[1] + (beta[3] * l_point[1])
    d_y_d_m = beta[2] + (beta[3] * l_point[0])

    J[0][0] = d_x_d_l
    J[0][1] = d_x_d_m
    J[1][0] = d_y_d_l
    J[1][1] = d_y_d_m

    return J

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
    # A \"numpy\" empty array (size == 0) of shape (0,).
    cdef numpy.ndarray[dtype = numpy.float64_t,
                       ndim = 1] n_e_array = \
         numpy.array([], \
                     dtype = numpy.float64)

    if (points.size == 0):
        return n_e_array

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
                   numpy.ndarray[dtype = numpy.float64_t, ndim = 2] p_t_m): #Perspective transformation matrix
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

def metric_coefficients(int dimension                                          ,
                        numpy.ndarray[dtype = numpy.float64_t, ndim = 1] point ,
                        numpy.ndarray[dtype = numpy.float64_t, ndim = 2] coeffs):
    cdef int dim = dimension
    cdef double den = 0.0
    cdef double num_epsilon_01 = 0.0
    cdef double num_epsilon_02 = 0.0
    cdef double num_nu_01 = 0.0
    cdef double num_nu_02 = 0.0
    cdef double den2 = 0.0
    cdef double d_epsilon_x = 0.0
    cdef double d_epsilon_y = 0.0
    cdef double d_nu_x = 0.0
    cdef double d_nu_y = 0.0
    # \"Numpy\" metric coefficients.
    cdef numpy.ndarray[dtype = numpy.float64_t, \
                       ndim = 2] n_m_cs = numpy.zeros(shape = (dim, \
                                                               dim), \
                                                      dtype = numpy.float64)

    add = numpy.add
    true_divide = numpy.true_divide
    square = numpy.square
    subtract = numpy.subtract
    multiply = numpy.multiply
    # A02*x + A12*y...
    den = add(numpy.multiply(point[0], coeffs[0][dim]), \
              numpy.multiply(point[1], coeffs[1][dim]))
    # ...+ A22.
    den = add(den, coeffs[dim][dim])
    # (A02*x + A12*y + A22)^2.
    den2 = square(den)
    # Part 01 of numerator for epsilon coordinate.
    num_epsilon_01 = den
    # Part 01 of numerator for nu coordinate.
    num_nu_01 = den
    # Part 02 of numerators for epsilon coordinate.
    # A00*x + A10*y...
    num_epsilon_02 = add(numpy.multiply(point[0], coeffs[0][0]), \
                         numpy.multiply(point[1], coeffs[1][0]))
    # ...+ A20
    num_epsilon_02 = add(num_epsilon_02, coeffs[dim][0])
    # Part 02 of numerators for nu coordinate.
    # A01*x + A11*y...
    num_nu_02 = add(numpy.multiply(point[0], coeffs[0][1]), \
                    numpy.multiply(point[1], coeffs[1][1]))
    # ...+ A21
    num_nu_02 = add(num_nu_02, coeffs[dim][1])
    # Derivative respect to \"epsilon\" for \"x\".
    d_epsilon_x = true_divide(subtract(multiply(num_epsilon_01,   \
                                                coeffs[0][0]),    \
                                       multiply(num_epsilon_02,   \
                                                coeffs[0][dim])), \
                              den2)
    # Derivative respect to \"epsilon\" for \"y\".
    d_epsilon_y = true_divide(subtract(multiply(num_epsilon_01,   \
                                                coeffs[1][0]),    \
                                       multiply(num_epsilon_02,   \
                                                coeffs[1][dim])), \
                              den2)
    # Derivative respect to \"nu\" for \"x\".
    d_nu_x = true_divide(subtract(multiply(num_nu_01,        \
                                           coeffs[0][1]),    \
                                  multiply(num_nu_02,        \
                                           coeffs[0][dim])), \
                         den2)
    # Derivative respect to \"nu\" for \"y\".
    d_nu_y = true_divide(subtract(multiply(num_nu_01,        \
                                           coeffs[1][1]),    \
                                  multiply(num_nu_02,        \
                                           coeffs[1][dim])), \
                         den2)
    # \"Numpy\" metric coefficients:
    # | \"d_epsilon_x\"    \"d_nu_x\" |
    # | \"d_epsilon_y\"    \"d_nu_y\" |
    n_m_cs[0][0] = d_epsilon_x
    n_m_cs[0][1] = d_nu_x
    n_m_cs[1][0] = d_epsilon_y
    n_m_cs[1][1] = d_nu_y

    return n_m_cs

#def metric_coefficients(dimension          ,
#                        in_points          ,
#                        matrix_coefficients,
#                        logger             ,
#                        log_file):
#    logger = check_null_logger(logger, log_file)
#    dim = dimension
#    A = matrix_coefficients
#    # http://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
#    if (type(in_points).__module__ == numpy.__name__):
#        points = in_points
#    else: 
#        points = numpy.array(in_points)
#    # Denominators.
#    # A02*x + A12*y...
#    dens = (numpy.add(numpy.multiply(points[:, 0], A[0][dim]), \
#                      numpy.multiply(points[:, 1], A[1][dim])))
#    # ...+ A22.
#    dens = numpy.add(dens, A[dim][dim])
#    # Part 01 of numerators for epsilon coordinate.
#    nums_epsilon_01 = dens 
#    # Part 01 of numerators for nu coordinate.
#    nums_nu_01 = dens 
#    # (A02*x + A12*y + A22)^2.
#    dens2 = numpy.square(dens)
#    # (A02*x + A12*y + A22)^4.
#    dens4 = numpy.square(dens2)
#    # Part 02 of numerators for epsilon coordinate.
#    # A00*x + A10*y...
#    nums_epsilon_02 = (numpy.add(numpy.multiply(points[:, 0], A[0][0]), \
#                                 numpy.multiply(points[:, 1], A[1][0])))
#    # ...+ A20
#    nums_epsilon_02 = numpy.add(nums_epsilon_02, A[dim][0])
#    # Part 02 of numerators for nu coordinate.
#    # A01*x + A11*y...
#    nums_nu_02 = (numpy.add(numpy.multiply(points[:, 0], A[0][1]), \
#                            numpy.multiply(points[:, 1], A[1][1])))
#    # ...+ A21
#    nums_nu_02 = numpy.add(nums_nu_02, A[dim][1])
#
#    ds_epsilon_x = numpy.true_divide(numpy.subtract(numpy.multiply(nums_epsilon_01, A[0][0]),
#                                                    numpy.multiply(nums_epsilon_02, A[0][dim])),
#                                     dens2)
#    ds_epsilon_y = numpy.true_divide(numpy.subtract(numpy.multiply(nums_epsilon_01, A[1][0]),
#                                                    numpy.multiply(nums_epsilon_02, A[1][dim])),
#                                     dens2)
#    ds_nu_x = numpy.true_divide(numpy.subtract(numpy.multiply(nums_nu_01, A[0][1]),
#                                               numpy.multiply(nums_nu_02, A[0][dim])),
#                                dens2)
#    ds_nu_y = numpy.true_divide(numpy.subtract(numpy.multiply(nums_nu_01, A[1][1]),
#                                               numpy.multiply(nums_nu_02, A[1][dim])),
#                                dens2)
#    ds2_epsilon_x = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[0][2],
#                                                                    dens),
#                                                     numpy.sum([numpy.multiply(A[0][0]*A[1][2] - A[0][2]*A[1][0],
#                                                                              points[:, 1]),
#                                                               (A[0][0]*A[2][2] - A[0][2]*A[2][0])])),
#                                      dens4)
#    ds2_epsilon_y = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[1][2],
#                                                                    dens),
#                                                     numpy.sum([numpy.multiply(A[1][0]*A[0][2] - A[1][2]*A[0][0],
#                                                                              points[:, 0]),
#                                                               (A[1][0]*A[2][2] - A[1][2]*A[2][0])])),
#                                      dens4)
#    ds2_nu_x = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[0][2],
#                                                               dens),
#                                                numpy.sum([numpy.multiply(A[0][1]*A[1][2] - A[0][2]*A[1][1],
#                                                                         points[:, 1]),
#                                                          (A[0][1]*A[2][2] - A[0][2]*A[2][1])])),
#                                 dens4)
#    ds2_nu_y = numpy.true_divide(numpy.multiply(numpy.multiply(-2*A[1][2],
#                                                               dens),
#                                                numpy.sum([numpy.multiply(A[1][1]*A[0][2] - A[1][2]*A[0][1],
#                                                                         points[:, 0]),
#                                                          (A[1][1]*A[2][2] - A[1][2]*A[2][1])])),
#                                      dens4)
#                                                                    
#    # Metric coefficients.
#    m_cs = [ds_epsilon_x, ds_epsilon_y, ds_nu_x, ds_nu_y, ds2_epsilon_x, ds2_epsilon_y, ds2_nu_x, ds2_nu_y]
#    # Numpy metric coefficients.
#    n_m_cs = numpy.array(m_cs)
#
#    return n_m_cs
       
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
    cdef double divisor = 0.0
    cdef double w_first

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
                      # Return also numpy data
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
