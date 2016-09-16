import unittest
from mpi4py import MPI
import project.main as main
import project.utilities as utilities
import ConfigParser
import os
import sys

log_file = "./log/mainTest.log"

class mainTest(unittest.TestCase):
    """Class which test some behaviours of the file \"project.main\".
    
       Attributes:
           comm_w (MPI.Intracomm) : Intracommunicator, copied by the one from
                                    the file to test.
           n_grids (int) : How many grids there are to be followed.
           rank_w (int) :  \"World\" rank of the current process.
           proc_grid (int) : Number defining the grid on which the current
                             process is working.
           procs_w (int) : How many processes are used to run the program,
                           in the \"comm_w\" intracommunicator.
           procs_l_lists (list[list]) : list containing the lists of processes
                                        for each grid.
           comm_l (petsc4py.MPI.Intracomm) : Intracommunicator for each grid.
           msg (string) : a message to be logged during some methods run.
           logger (utilities.Logger) : A logger."""
           
    def setUp(self):
        self.n_grids = main.n_grids
        self.comm_w = main.comm_w
        self.rank_w = self.comm_w.Get_rank()
        group_w = self.comm_w.Get_group()
        self.procs_w = self.comm_w.Get_size()
        procs_w_list = range(0, self.procs_w)
        self.procs_l_lists = utilities.chunk_list_ordered(procs_w_list,
                                                          self.n_grids)
        self.proc_grid =  utilities.get_proc_grid(self.procs_l_lists,
                                                  self.comm_w.Get_rank())
        group_l = group_w.Incl(self.procs_l_lists[self.proc_grid])
        self.comm_l = self.comm_w.Create(group_l)
        comm_name = main.comm_names[self.proc_grid]
        self.comm_l.Set_name(comm_name)
        self.msg = "Started function for local comm \"" + \
                   str(self.comm_l.Get_name())          + \
                   "\" and world comm \""               + \
                   str(self.comm_w.Get_name())          + \
                   "\" and rank \""                     + \
                   str(self.comm_l.Get_rank()) + "\"."
        self.logger = utilities.Logger(__name__, 
                                       log_file).logger

    def test_create_intercomms(self):
        """Method which tests that the number of intercommunicators
           created is equal to the total number of grids minus 1, for
           each grid."""

        self.logger.info(self.msg)
        
        intercomm_dictionary = {}
        if self.procs_w > 1:
            n_intercomms = self.n_grids - 1
            main.create_intercomms(self.n_grids      ,
                                   self.proc_grid    ,
                                   self.comm_l       ,
                                   self.procs_l_lists,
                                   self.logger       ,
                                   intercomm_dictionary)
            self.assertEqual(len(intercomm_dictionary), self.n_grids - 1)
        else:
            print("Called with only 1 MPI process.")
            self.assertDictEqual(intercomm_dictionary, {})

    def test_set_comm_dict(self):
        """Method which tests that the number of couple keys-values inserted
           into the dictionary by the method \"main.set_comm_dict\" is equal to 
           9 (as they should be)."""

        self.logger.info(self.msg)

        comm_dict = main.set_comm_dict(self.n_grids  ,
                                       self.proc_grid,
                                       self.comm_l)
        self.assertTrue(len(comm_dict) == 11)

    def test_set_octree(self):
        """Method which tests that the length of the list containing the
           centers of the quadtree is equal to the number of the octants
           (aka quadtree in 2D) returned by the octree."""

        self.logger.info(self.msg)

        pablo, centers = main.set_octree(self.comm_l,
                                         self.proc_grid)
        self.assertTrue(len(centers) == pablo.get_num_octants())

    def test_main(self):
        """Method which just see if the call to \"main.main\" is going without
           problems inside the called function."""

        self.logger.info(self.msg)

        main.log_file = log_file
        main.main()


    def tearDown(self):
        try:
            del self.n_grids
        except NameError:
            sys.exc_info()[1]
            print("Attribute not defined.")
        finally:
            del self.comm_w
            del self.rank_w
            del self.proc_grid
            del self.procs_w
            del self.procs_l_lists
            del self.comm_l
            del self.msg
            del self.logger

if __name__ == "__main__":
    if os.path.exists(log_file):
        with open(log_file, "w") as of:
            pass
    suite = unittest.TestLoader().loadTestsFromTestCase(mainTest)
    unittest.TextTestRunner().run(suite)
