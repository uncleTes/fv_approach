# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import xml.etree.cElementTree as ET
import logging
import os
import sys
from mpi4py import MPI
import numpy 
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

def points_location(point_to_check,
                    other_point):
    if ((point_to_check[0] - other_point[0] <= 0) and
        (point_to_check[1] - other_point[1] >= 0)):
        return "nordovest"
    if ((point_to_check[0] - other_point[0] > 0) and
        (point_to_check[1] - other_point[1] >= 0)):
        return "nordest"
    if ((point_to_check[0] - other_point[0] <= 0) and
        (point_to_check[1] - other_point[1] < 0)):
        return "sudovest"
    if ((point_to_check[0] - other_point[0] > 0) and
        (point_to_check[1] - other_point[1] < 0)):
        return "sudest"

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

def check_into_circle(point_to_check,
                      circle_center ,
                      circle_radius):
    check = False

    distance2 = math.pow((point_to_check[0] - circle_center[0]) , 2) + \
                math.pow((point_to_check[1] - circle_center[1]) , 2)
    distance = math.sqrt(distance2)

    if distance <= circle_radius:
        check = True

    return check

def get_corners_from_center(center,
                            edge):
    corners = []
    corners.append([center[0] - (edge/2.0), center[1] - (edge/2.0)])
    corners.append([center[0] + (edge/2.0), center[1] - (edge/2.0)])
    corners.append([center[0] - (edge/2.0), center[1] + (edge/2.0)])
    corners.append([center[0] + (edge/2.0), center[1] + (edge/2.0)])

    return corners

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
            # TODO: understand where to put the threshold.
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

def check_point_into_square(point_to_check,
                            # [x_anchor, x_anchor + edge, 
                            #  y_anchor, y_anchor + edge]
                            square        ,
                            threshold     ,
                            logger        ,
                            log_file):
    check = False

    if isinstance(square, list):
        if ((point_to_check[0] - square[0] >= threshold) and
            (square[1] - point_to_check[0] >= threshold) and
            (point_to_check[1] - square[2] >= threshold) and
            (square[3] - point_to_check[1] >= threshold)):
            check = True
    else:
        logger = check_null_logger(logger, 
                                   log_file)
        logger.error("Second parameter must be a list.")
    return check

def check_oct_into_square(oct_center,
                          square    ,
                          oct_edge  ,
                          threshold ,
                          logger    ,
                          log_file):
    check = False
    points_to_check = []
    an00 = (oct_center[0] - (oct_edge / 2),
            oct_center[1] - (oct_edge / 2))
    an10 = (oct_center[0] + (oct_edge / 2),
            oct_center[1] - (oct_edge / 2))
    an01 = (oct_center[0] - (oct_edge / 2),
            oct_center[1] + (oct_edge / 2))
    an11 = (oct_center[0] + (oct_edge / 2),
            oct_center[1] + (oct_edge / 2))
    points_to_check.append(an00)
    points_to_check.append(an10)
    points_to_check.append(an01)
    points_to_check.append(an11)

    for i, point in enumerate(points_to_check):
        check = check_point_into_square(point    ,
                                        square   ,
                                        threshold,
                                        logger   ,
                                        log_file)
        if not check:
            break

    return check

def check_oct_into_squares(oct_center,
                           squares   ,
                           oct_edge  ,
                           threshold ,
                           logger    ,
                           log_file):
    check = False
    points_to_check = []
    an00 = (oct_center[0] - (oct_edge / 2),
            oct_center[1] - (oct_edge / 2))
    an10 = (oct_center[0] + (oct_edge / 2),
            oct_center[1] - (oct_edge / 2))
    an01 = (oct_center[0] - (oct_edge / 2),
            oct_center[1] + (oct_edge / 2))
    an11 = (oct_center[0] + (oct_edge / 2),
            oct_center[1] + (oct_edge / 2))
    points_to_check.append(an00)
    points_to_check.append(an10)
    points_to_check.append(an01)
    points_to_check.append(an11)

    for i, point in enumerate(points_to_check):
        check = check_point_into_squares(point    ,
                                         squares  ,
                                         threshold,
                                         logger   ,
                                         log_file)
        if not check:
            break

    return check

def check_point_into_squares(point_to_check, 
                             # [[x_anchor, x_anchor + edge, 
                             #   y_anchor, y_anchor + edge]...]
                             squares       ,
                             threshold     ,
                             logger        ,
                             log_file):
    check = False

    if isinstance(squares, list):
        for i, square in enumerate(squares):
            check = check_point_into_square(point_to_check,
                                            square        ,
                                            threshold     ,
                                            logger        ,
                                            log_file)
            if check:
                return check
    else:
        logger = check_null_logger(logger, 
                                   log_file)
        logger.error("Second parameter must be a list of lists.")
    return check

# https://it.wikipedia.org/wiki/Metodo_dei_minimi_quadrati
def least_squares(points,
                  unknown_point):
    # In 2D we approximate our function as a plane: \"ax + by + c\".
    n_points = len(points)
    A = numpy.zeros((n_points, 3))

    for i in xrange(0, n_points):
        t_point = points[i]
        if (type(t_point) is list):
            # Append to the list the coefficient for the \"c\" parameter.
            t_point.append(1)
        elif (type(t_point) is tuple):
            # Adding to the tuple the coefficient for the \"c\" parameter.
            t_point = t_point + (1,)
        A[i] = t_point

    At = A.T
    AtA = numpy.dot(At, A)
    # Pseudo-inverse matrix.
    p = numpy.dot(numpy.linalg.inv(AtA), At)
    # Multiplying \"a\" time \"x\".
    p[0, :] = p[0, :] * unknown_point[0]
    # Multiplying \"b\" time \"y\".
    p[1, :] = p[1, :] * unknown_point[1]

    coeffs = numpy.sum(p, axis = 0)
    return coeffs
    
def bil_coeffs(unknown_point, 
               points_coordinates):
    coeff_01 = ((points_coordinates[3][0] - unknown_point[0]) * 
                (points_coordinates[3][1] - unknown_point[1]))

    coeff_02 = ((unknown_point[0] - points_coordinates[0][0]) *
                (points_coordinates[3][1] - unknown_point[1]))

    coeff_03 = ((points_coordinates[3][0] - unknown_point[0]) *
                (unknown_point[1] - points_coordinates[0][1]))

    coeff_04 = ((unknown_point[0] - points_coordinates[0][0]) *
                (unknown_point[1] - points_coordinates[0][1]))

    multiplier = 1 / ((points_coordinates[3][0] - points_coordinates[0][0]) * 
                      (points_coordinates[3][1] - points_coordinates[0][1]))

    coefficients = [coeff_01, coeff_02, coeff_03, coeff_04]

    coefficients = [coefficient * multiplier for coefficient in coefficients]

    return coefficients

# http://en.wikipedia.org/wiki/Bilinear_interpolation
#   Q12------------Q22
#      |          |
#      |          |
#      |          |
#      |          |
#      |      x,y |
#   Q11-----------Q21
#   Q11 = point_values at x1 and y1
#   Q12 = point_values at x1 and y2
#   Q21 = point_values at x2 and y1
#   Q22 = point_values at x2 and y2
#   f(Q11) = value of the function in x1 and y1
#   f(Q12) = value of the function in x1 and y2
#   f(Q21) = value of the function in x2 and y1
#   f(Q22) = value of the function in x2 and y2
#   x,y = unknown_point ("unknown point" stand for a point for which it is 
#         not known the value of the function f)
def bil_interp(unknown_point	 , 
               points_coordinates,
               f_values):
    coeff_01 = ((points_coordinates[3][0] - unknown_point[0]) * 
                (points_coordinates[3][1] - unknown_point[1]))

    coeff_02 = ((unknown_point[0] - points_coordinates[0][0]) *
                (points_coordinates[3][1] - unknown_point[1]))

    coeff_03 = ((points_coordinates[3][0] - unknown_point[0]) *
                (unknown_point[1] - points_coordinates[0][1]))

    coeff_04 = ((unknown_point[0] - points_coordinates[0][0]) *
                (unknown_point[1] - points_coordinates[0][1]))

    multiplier = 1 / ((points_coordinates[3][0] - points_coordinates[0][0]) * 
                      (points_coordinates[3][1] - points_coordinates[0][1]))

    coefficients = [coeff_01, coeff_02, coeff_03, coeff_04]

    coefficients = [coefficient * multiplier for coefficient in coefficients]
    
    value = 0

    for i, coeff in enumerate(coefficients):
        value += (coeff * point_f_values[i])

    return value

# Perspective transformation coefficients (linear coefficients).
def p_t_coeffs(dim               ,
               original_points   ,
               transformed_points,
               logger            ,
               log_file):
    coefficients = None
    matrix = None
    rhs = None
    logger = check_null_logger(logger, log_file)

    try:
        assert isinstance(original_points, 
                          list), "No list passed as third argument." 
        assert isinstance(transformed_points,
                          list), "No list passed as fourth argument."
        assert (2 <= dim <= 3), "Wrong dimension passed as first parameter."

        if (dim == 2):
            uv = numpy.array(original_points, dtype = numpy.float64)
            uv.resize(4, 3)
            xy = numpy.array(transformed_points, dtype = numpy.float64)
            xy.resize(4,3)
            matrix = numpy.array([[uv[0][0], uv[0][1], 1, 0, 0, 0, -(uv[0][0] * xy[0][0]), -(uv[0][1] * xy[0][0])],
                                  [uv[1][0], uv[1][1], 1, 0, 0, 0, -(uv[1][0] * xy[1][0]), -(uv[1][1] * xy[1][0])],
                                  [uv[2][0], uv[2][1], 1, 0, 0, 0, -(uv[2][0] * xy[2][0]), -(uv[2][1] * xy[2][0])],
                                  [uv[3][0], uv[3][1], 1, 0, 0, 0, -(uv[3][0] * xy[3][0]), -(uv[3][1] * xy[3][0])],
                                  [0, 0, 0, uv[0][0], uv[0][1], 1, -(uv[0][0] * xy[0][1]), -(uv[0][1] * xy[0][1])],
                                  [0, 0, 0, uv[1][0], uv[1][1], 1, -(uv[1][0] * xy[1][1]), -(uv[1][1] * xy[1][1])],
                                  [0, 0, 0, uv[2][0], uv[2][1], 1, -(uv[2][0] * xy[2][1]), -(uv[2][1] * xy[2][1])],
                                  [0, 0, 0, uv[3][0], uv[3][1], 1, -(uv[3][0] * xy[3][1]), -(uv[3][1] * xy[3][1])]])
            rhs = numpy.array([xy[0][0], xy[1][0], xy[2][0], xy[3][0], xy[0][1], xy[1][1], xy[2][1], xy[3][1]])
        # Dim = 3.
        else:
            uvw = original_points
            xyz = transformed_points
            matrix = numpy.array([[uvw[0][0], uvw[0][1], uvw[0][2], 1, 0, 0, 0, 0, 0, 0, 0, 0, -(uvw[0][0]*xyz[0][0]), -(uvw[0][1]*xyz[0][0]), -(uvw[0][2]*xyz[0][0])],
                                  [uvw[1][0], uvw[1][1], uvw[1][2], 1, 0, 0, 0, 0, 0, 0, 0, 0, -(uvw[1][0]*xyz[1][0]), -(uvw[1][1]*xyz[1][0]), -(uvw[1][2]*xyz[1][0])],
                                  [uvw[2][0], uvw[2][1], uvw[2][2], 1, 0, 0, 0, 0, 0, 0, 0, 0, -(uvw[2][0]*xyz[2][0]), -(uvw[2][1]*xyz[2][0]), -(uvw[2][2]*xyz[2][0])],
                                  [uvw[3][0], uvw[3][1], uvw[3][2], 1, 0, 0, 0, 0, 0, 0, 0, 0, -(uvw[3][0]*xyz[3][0]), -(uvw[3][1]*xyz[3][0]), -(uvw[3][2]*xyz[3][0])],
                                  [0, 0, 0, 0, uvw[0][0], uvw[0][1], uvw[0][2], 1, 0, 0, 0, 0, -(uvw[0][0]*xyz[0][1]), -(uvw[0][1]*xyz[0][1]), -(uvw[0][2]*xyz[0][1])],
                                  [0, 0, 0, 0, uvw[1][0], uvw[1][1], uvw[1][2], 1, 0, 0, 0, 0, -(uvw[1][0]*xyz[1][1]), -(uvw[1][1]*xyz[1][1]), -(uvw[1][2]*xyz[1][1])],
                                  [0, 0, 0, 0, uvw[2][0], uvw[2][1], uvw[2][2], 1, 0, 0, 0, 0, -(uvw[2][0]*xyz[2][1]), -(uvw[2][1]*xyz[2][1]), -(uvw[2][2]*xyz[2][1])],
                                  [0, 0, 0, 0, uvw[3][0], uvw[3][1], uvw[3][2], 1, 0, 0, 0, 0, -(uvw[3][0]*xyz[3][1]), -(uvw[3][1]*xyz[3][1]), -(uvw[3][2]*xyz[3][1])],
                                  [0, 0, 0, 0, 0, 0, 0, 0, uvw[0][0], uvw[0][1], uvw[0][2], 1, -(uvw[0][0]*xyz[0][2]), -(uvw[0][1]*xyz[0][2]), -(uvw[0][2]*xyz[0][2])],
                                  [0, 0, 0, 0, 0, 0, 0, 0, uvw[1][0], uvw[1][1], uvw[1][2], 1, -(uvw[1][0]*xyz[1][2]), -(uvw[1][1]*xyz[1][2]), -(uvw[1][2]*xyz[1][2])],
                                  [0, 0, 0, 0, 0, 0, 0, 0, uvw[2][0], uvw[2][1], uvw[2][2], 1, -(uvw[2][0]*xyz[2][2]), -(uvw[2][1]*xyz[2][2]), -(uvw[2][2]*xyz[2][2])],
                                  [0, 0, 0, 0, 0, 0, 0, 0, uvw[3][0], uvw[3][1], uvw[3][2], 1, -(uvw[3][0]*xyz[3][2]), -(uvw[3][1]*xyz[3][2]), -(uvw[3][2]*xyz[3][2])]])
            
            rhs = numpy.array(xyz[0][0], xyz[1][0], xyz[2][0], xyz[3][0], xyz[0][1], xyz[1][1], xyz[2][1], xyz[3][1], xyz[0][2], xyz[1][2], xyz[2][2], xyz[3][2])
        
        coefficients = numpy.linalg.solve(matrix, rhs)
        # \"append\" does not occur in place:
        # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.append.html
        # We append 1 as coefficients \"a33\" (or \"a44\", depending on problem's 
        # dimension) without loss of generality.
        coefficients = numpy.append(coefficients, 1)
        coefficients = coefficients.reshape(dim + 1, dim + 1).T    
    except AssertionError:
        msg_err = sys.exc_info()[1] 
        logger.error(msg_err)
    finally:
        return coefficients

# Perspective transformation coefficients for adjoint matrix.
def p_t_coeffs_adj(dim       ,
                   p_t_coeffs,
                   logger    ,
                   log_file):
    # Adjoint matrix
    ad_matrix = None
    A = p_t_coeffs
    A_det = numpy.linalg.det(A)
    logger = check_null_logger(logger, log_file)
    
    try:
        assert (2 <= dim <= 3), "Wrong dimension passed as first parameter."

        if (dim == 2):
            ad_matrix = numpy.array([[(A[1][1] * A[2][2]) - (A[1][2] * A[2][1]),
                                      (A[0][2] * A[2][1]) - (A[0][1] * A[2][2]),
                                      (A[0][1] * A[1][2]) - (A[0][2] * A[1][1]),
                                     ],
                                     [(A[1][2] * A[2][0]) - (A[1][0] * A[2][2]),
                                      (A[0][0] * A[2][2]) - (A[0][2] * A[2][0]),
                                      (A[0][2] * A[1][0]) - (A[0][0] * A[1][2]),
                                     ],
                                     [(A[1][0] * A[2][1]) - (A[1][1] * A[2][0]),
                                      (A[0][1] * A[2][0]) - (A[0][0] * A[2][1]),
                                      (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]),
                                     ]])
        # Dim = 3.
        else:
            ad00 = (A[1][2] * A[2][3] * A[3][1]) - (A[1][3] * A[2][2] * A[3][1]) + (A[1][3] * A[2][1] * A[3][2]) - \
                   (A[1][1] * A[2][3] * A[3][2]) - (A[1][2] * A[2][1] * A[3][3]) + (A[1][1] * A[2][2] * A[3][3])
            ad01 = (A[0][3] * A[2][2] * A[3][1]) - (A[0][2] * A[2][3] * A[3][1]) - (A[0][3] * A[2][1] * A[3][2]) + \
                   (A[0][1] * A[2][3] * A[3][2]) + (A[0][2] * A[2][1] * A[3][3]) - (A[0][1] * A[2][2] * A[3][3])
            ad02 = (A[0][2] * A[1][3] * A[3][1]) - (A[0][3] * A[1][2] * A[3][1]) + (A[0][3] * A[1][1] * A[3][2]) - \
                   (A[0][1] * A[1][3] * A[3][2]) - (A[0][2] * A[1][1] * A[3][3]) + (A[0][1] * A[1][2] * A[3][3])
            ad03 = (A[0][3] * A[1][2] * A[2][1]) - (A[0][2] * A[1][3] * A[2][1]) - (A[0][3] * A[1][1] * A[2][2]) + \
                   (A[0][1] * A[1][3] * A[2][2]) + (A[0][2] * A[1][1] * A[2][3]) - (A[0][1] * A[1][2] * A[2][3])
            ad10 = (A[1][3] * A[2][2] * A[3][0]) - (A[1][2] * A[2][3] * A[3][0]) - (A[1][3] * A[2][0] * A[3][2]) + \
                   (A[1][0] * A[2][3] * A[3][2]) + (A[1][2] * A[2][0] * A[3][3]) - (A[1][0] * A[2][2] * A[3][3])
            ad11 = (A[0][2] * A[2][3] * A[3][0]) - (A[0][3] * A[2][2] * A[3][0]) + (A[0][3] * A[2][0] * A[3][2]) - \
                   (A[0][0] * A[2][3] * A[3][2]) - (A[0][2] * A[2][0] * A[3][3]) + (A[0][0] * A[2][2] * A[3][3])
            ad12 = (A[0][3] * A[1][2] * A[3][0]) - (A[0][2] * A[1][3] * A[3][0]) - (A[0][3] * A[1][0] * A[3][2]) + \
                   (A[0][0] * A[1][3] * A[3][2]) + (A[0][2] * A[1][0] * A[3][3]) - (A[0][0] * A[1][2] * A[3][3])
            ad13 = (A[0][2] * A[1][3] * A[2][0]) - (A[0][3] * A[1][2] * A[2][0]) + (A[0][3] * A[1][0] * A[2][2]) - \
                   (A[0][0] * A[1][3] * A[2][2]) - (A[0][2] * A[1][0] * A[2][3]) + (A[0][0] * A[1][2] * A[2][3])
            ad20 = (A[1][1] * A[2][3] * A[3][0]) - (A[1][3] * A[2][1] * A[3][0]) + (A[1][3] * A[2][0] * A[3][1]) - \
                   (A[1][0] * A[2][3] * A[3][1]) - (A[1][1] * A[2][0] * A[3][3]) + (A[1][0] * A[2][1] * A[3][3])
            ad21 = (A[0][3] * A[2][1] * A[3][0]) - (A[0][1] * A[2][3] * A[3][0]) - (A[0][3] * A[2][0] * A[3][1]) + \
                   (A[0][0] * A[2][3] * A[3][1]) + (A[0][1] * A[2][0] * A[3][3]) - (A[0][0] * A[2][1] * A[3][3])
            ad22 = (A[0][1] * A[1][3] * A[3][0]) - (A[0][3] * A[1][1] * A[3][0]) + (A[0][3] * A[1][0] * A[3][1]) - \
                   (A[0][0] * A[1][3] * A[3][1]) - (A[0][1] * A[1][0] * A[3][3]) + (A[0][0] * A[1][1] * A[3][3])
            ad23 = (A[0][3] * A[1][1] * A[2][0]) - (A[0][1] * A[1][3] * A[2][0]) - (A[0][3] * A[1][0] * A[2][1]) + \
                   (A[0][0] * A[1][3] * A[2][1]) + (A[0][1] * A[1][0] * A[2][3]) - (A[0][0] * A[1][1] * A[2][3])
            ad30 = (A[1][2] * A[2][1] * A[3][0]) - (A[1][1] * A[2][2] * A[3][0]) - (A[1][2] * A[2][0] * A[3][1]) + \
                   (A[1][0] * A[2][2] * A[3][1]) + (A[1][1] * A[2][0] * A[3][2]) - (A[1][0] * A[2][1] * A[3][2])
            ad31 = (A[0][1] * A[2][2] * A[3][0]) - (A[0][2] * A[2][1] * A[3][0]) + (A[0][2] * A[2][0] * A[3][1]) - \
                   (A[0][0] * A[2][2] * A[3][1]) - (A[0][1] * A[2][0] * A[3][2]) + (A[0][0] * A[2][1] * A[3][2])
            ad32 = (A[0][2] * A[1][1] * A[3][0]) - (A[0][1] * A[1][2] * A[3][0]) - (A[0][2] * A[1][0] * A[3][1]) + \
                   (A[0][0] * A[1][2] * A[3][1]) + (A[0][1] * A[1][0] * A[3][2]) - (A[0][0] * A[1][1] * A[3][2])
            ad33 = (A[0][1] * A[1][2] * A[2][0]) - (A[0][2] * A[1][1] * A[2][0]) + (A[0][2] * A[1][0] * A[2][1]) - \
                   (A[0][0] * A[1][2] * A[2][1]) - (A[0][1] * A[1][0] * A[2][2]) + (A[0][0] * A[1][1] * A[2][2])

            ad_matrix = numpy.array([[ad00, ad01, ad02, ad03],
                                     [ad10, ad11, ad12, ad13],
                                     [ad20, ad21, ad22, ad23],
                                     [ad30, ad31, ad32, ad33]])
    except AssertionError:
        msg_err = sys.exc_info()[1] 
        logger.error(msg_err)
    finally:
        ad_matrix = numpy.true_divide(ad_matrix, A_det)
        return ad_matrix

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
       
# Homogeneous coordinate w'.
def h_c_w_first(dimension   ,
                in_points   ,
                coefficients,
                logger      ,
                log_file):
    logger = check_null_logger(logger, log_file)
    dim = dimension
    # http://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    if (type(in_points).__module__ == numpy.__name__):
        points = in_points
    else: 
        points = numpy.array(in_points)
    x_s = numpy.multiply(points[:, 0], coefficients[0][dim])
    y_s = numpy.multiply(points[:, 1], coefficients[1][dim])
    if (dim == 3):
        z_s = numpy.multiply(points[:, 2], coefficients[2][dim])
    # Inverse of w's (made to be done for vectors).
    i_w_s_first = numpy.add(x_s, y_s)
    if (dim == 3):
        i_w_s_first = numpy.add(i_w_s_first, z_s)
    i_w_s_first = numpy.add(i_w_s_first, coefficients[dim][dim])
    w_s_first = numpy.true_divide(1.0, i_w_s_first)
    if (w_s_first.shape[0] == 1):
        return w_s_first[0]
    else:
        return w_s_first

def compute_w_first(logger,
                    point,
                    coefficients):
    w_first = None
    dim = len(point)
    ad_matrix = coefficients

    try:
        assert (2 <= dim <= 3), "Wrong size for the array passed as point."
        if (dim == 2):
            xy = point
            w_first = 1.0 / ((ad_matrix[0][2] * xy[0]) + \
                           (ad_matrix[1][2] * xy[1]) + \
                           ad_matrix[2][2])
        else:
            xyz = point
            w_first = 1.0 / ((ad_matrix[0][3] * xyz[0]) + \
                           (ad_matrix[1][3] * xyz[1]) + \
                           (ad_matrix[2][3] * xyz[2]) + \
                           ad_matrix[3][3])

    except AssertionError:
        msg_err = sys.exc_info()[1] 
        logger.error(msg_err)
    finally:
        return w_first 


def apply_persp_trans_inv(dimension   ,
                          point       ,
                          coefficients,
                          logger      ,
                          log_file):
    # Numpy point.
    np_point = numpy.array(point[0 : dimension],
                           dtype = numpy.float64)
    dim = np_point.shape[0]
    # Homogeneous coordinates.
    np_point = numpy.append(np_point, 1)
    # Transformed inverse point.
    t_i_point = None
    logger = check_null_logger(logger, log_file)
    ad_matrix = coefficients

    try:
        assert (2 <= dim <= 3), "Wrong size for the array passed as point."
        if (dim == 2):
            xy = point
            w_first = 1 / ((ad_matrix[0][2] * xy[0]) + \
                           (ad_matrix[1][2] * xy[1]) + \
                           ad_matrix[2][2])
        else:
            xyz = point
            w_first = 1 / ((ad_matrix[0][3] * xyz[0]) + \
                           (ad_matrix[1][3] * xyz[1]) + \
                           (ad_matrix[2][3] * xyz[2]) + \
                           ad_matrix[3][3])

        np_point = numpy.multiply(np_point, w_first)
        # Numpy transformed inverse point.
        np_t_i_point = numpy.dot(np_point, ad_matrix)
        if (dimension == 2):
            # Returning however 3 coordinates, also being in 2D. New \"PABLO\"
            # is intrinsically 3D.
            np_t_i_point[-1] = 0.0
        else:
            np_t_i_point = np_t_i_point[0 : -1]
        t_i_point = np_t_i_point.tolist()
    except AssertionError:
        msg_err = sys.exc_info()[1] 
        logger.error(msg_err)
    finally:
        return t_i_point

def apply_persp_trans(dimension   ,
                      point       ,
                      coefficients,
                      logger      ,
                      log_file):
    # Numpy point.
    # http://stackoverflow.com/questions/14415741/numpy-array-vs-asarray
    np_point = numpy.array(point[0 : dimension], 
                           dtype = numpy.float64)
    # Homogeneous coordinates.
    np_point = numpy.append(np_point, 1)
    # Transformed point.
    t_point = None
    if log_file is not None:
        logger = check_null_logger(logger, log_file)
    try:
        # Number of columns equal to number of rows.
        assert (np_point.shape[0] == coefficients.shape[0]), \
               "Wrong dimensions for array-matrix multiplications."
        # Numpy transformed point.
        np_t_point = numpy.dot(np_point, coefficients)
        np_t_point = numpy.true_divide(np_t_point,
                                       np_t_point[-1])
        if (dimension == 2):
            # Returning however 3 coordinates, also being in 2D. New \"PABLO\"
            # is intrinsically 3D.
            np_t_point[-1] = 0.0 
        else:
            np_t_point = np_t_point[0 : -1]
        t_point = np_t_point.tolist()
    except AssertionError:
        msg_err = sys.exc_info()[1]
        if log_file is not None:
            logger.error(msg_err)
    finally:
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
