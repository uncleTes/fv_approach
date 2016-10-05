# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
from   mpi4py import MPI
import my_class_vtk
import my_pablo_uniform
import numpy

comm_w = MPI.COMM_WORLD
curr_proc = comm_w.Get_rank()
pablo_log_file = "./log/" + "prova_normal.log"
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
    pablo.set_balance(idx, False)
    # Computing connectivity/
    pablo.compute_connectivity()
    # Refining globally.
    pablo.adapt_global_refine()
    pablo.update_connectivity()
    # Balancing the octree among processes.
    pablo.load_balance()
    # Setting node 0 to be refined 1 time.
    pablo.set_marker(idx, 1)
    # Adapting the octree.
    pablo.adapt()
    # Balancing the octree among processes.
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

    for i in xrange(0, n_inter):
        inter = pablo.get_intersection(i)
        is_bound_inter = pablo.get_bound(inter, 
                                         0    , # Being with an intersection, it 
                                         True)  # does not matter what number we
                                                # are giving to the second arg.
        owners_inter = pablo.get_owners(inter)
        # Global indeces of the owners.
        owners_g_inter, owners_g_level = ([] for i in range(0, 2))
        owner_g = pablo.get_global_idx(owners_inter[0])
        owner_g_oct = pablo.get_octant(owners_inter[0])
        # Level of the global owner.
        owner_g_l = pablo.get_level(owner_g_oct,
                                    True) # Using an octant instead of an index.
        owners_g_inter.append(owner_g)
        owners_g_level.append(owner_g_l)
        is_ghost_inter = pablo.get_is_ghost(inter, 
                                            True) # Using intersection instead
                                                  # of octant.
        if (is_ghost_inter):
            owner_g = pablo.get_ghost_global_idx(owners_inter[1])
            owner_g_oct = pablo.get_ghost_octant(owners_inter[1])
        else:
            owner_g = pablo.get_global_idx(owners_inter[1])
            owner_g_oct = pablo.get_octant(owners_inter[1])

        owners_g_inter.append(owner_g)
        owner_g_l = pablo.get_level(owner_g_oct,
                                    True)
        owners_g_level.append(owner_g_l)
        
        finer_inter = int(pablo.get_finer(inter))
        normal_inter = pablo.get_normal(inter) 

        to_print = join_strings("Rank "              , 
                                str(curr_proc)       ,
                                ": intersection "    ,
                                str(i)               ,
                                " is of border: "    ,
                                str(is_bound_inter)  ,
                                " is ghost: "        ,
                                str(is_ghost_inter)  ,
                                " owners: "          ,
                                str(owners_inter)    ,
                                " global owners: "   ,
                                str(owners_g_inter)  ,
                                " normal: "          ,
                                str(normal_inter)    ,
                                " finer owner: "     ,
                                str(finer_inter)     ,
                                " levels of owners: ",
                                str(owners_g_level))

        print(to_print)
    


if __name__ == "__main__":
    main()
