# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
from   mpi4py import MPI
import my_class_vtk
import my_pablo_uniform
import numpy

comm_w = MPI.COMM_WORLD
curr_proc = comm_w.Get_rank()
pablo_log_file = "./log/" + "prova_adapt.log"
dim = 2

def join_strings(*args):
    # List of strings to join.
    strs_list = []
    map(strs_list.append, args)
    # Returned string.
    r_s = "".join(strs_list)
    return r_s

def main():    
    pablo = my_pablo_uniform.Py_My_Pablo_Uniform(0             , # anchor x
                                                 0             , # anchor y
                                                 0             , # anchor z
                                                 1             , # edge
                                                 dim           , # 2D/3D
                                                 20            , # Max level
                                                 pablo_log_file, # Logfile
                                                 comm_w)         # Comm

    idx = 0
    # Balance 2:1 is False.
    pablo.set_balance(idx, True)
    for i in xrange(0, 2):
        # Refining globally.
        pablo.adapt_global_refine()
    n_octs = pablo.get_num_octants()

    idx_lim = ((n_octs * 3) / 4)
    #idx_lim = 0

    for idx in xrange(0, n_octs):
        if (idx < idx_lim):
            pablo.set_marker(idx, 1)
    pablo.adapt()
    pablo.load_balance()
    pablo.update_connectivity()
    # Computing new intersections.
    pablo.compute_intersections()

    n_inter = pablo.get_num_intersections()
    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    octants = range(0, n_octs)
    data_to_save = numpy.array([octants, octants], 
                               dtype = numpy.float)

    vtk = my_class_vtk.Py_My_Class_VTK(data_to_save            , # Data
                                       pablo                   , # Octree
                                       "./data/"               , # Dir
                                       "prova_normal_intersect", # Name
                                       "ascii"                 , # Type
                                       n_octs                  , # Ncells
                                       n_nodes                 , # Nnodes
                                       (2**dim) * n_octs)        # (Nnodes * 
                                                                 #  pow(2,dim))
    ## Add data to "vtk" object to be written later.
    vtk.add_data("evaluated", # Data
                 1          , # Data dim
                 "Float64"  , # Data type
                 "Cell"     , # Cell or Point
                 "ascii")     # File type
    vtk.add_data("exact"  , 
                 1        , 
                 "Float64", 
                 "Cell"   , 
                 "ascii")

    # Call parallelization and writing onto file.
    vtk.print_vtk()

if __name__ == "__main__":
    main()
