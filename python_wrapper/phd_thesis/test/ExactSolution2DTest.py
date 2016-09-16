import unittest
from mpi4py import MPI
import numpy
import os
# http://docs.python-guide.org/en/latest/writing/structure/
# http://stackoverflow.com/questions/13621540/import-a-file-from-different-directory
import project.ExactSolution2D as ExactSolution2D
import class_para_tree

log_file = "./log/ExactSolution2DTest.log"

class ExactSolution2DTest(unittest.TestCase):
    """Class which test some behaviours of the class \"ExactSolution2D\".
    
       Attributes:
           comm (MPI.Intracomm) : test intracommunicator.
           pablo (class_para_tree.Py_Class_Para_Tree_D2) : test octree.
           comm_dictionary (dictionary) : test dictionary.
           centers (list[lists]) : list of coordinates of cells' centers."""

    # Deriving from \"unittest.TestCase\", this class can overload this 
    # method to prepare the environment before each test contained in the class.
    def setUp(self):
        # Initialize MPI.
        self.comm = MPI.COMM_WORLD
        self.comm_dictionary = {}
        self.comm_dictionary.update({"log file" : log_file})
        # Octree
        self.pablo = class_para_tree.Py_Class_Para_Tree_D2(0.0     ,
                                                           0.0     ,
                                                           0.0     ,
                                                           1.0     ,
                                                           log_file,  # Logfile
                                                           self.comm) # Comm
        self.pablo.set_balance(0, True)
        
        for iteration in xrange(1, 3):
            self.pablo.adapt_global_refine()
        
        self.pablo.load_balance()
        self.pablo.update_connectivity()
        self.pablo.update_ghosts_connectivity()
        
        n_octs = self.pablo.get_num_octants()
        
        self.centers = numpy.empty([n_octs, 2])

        for i in xrange(0, n_octs):
            g_idx = self.pablo.get_global_idx(i)
            # Getting fields 0 and 1 of \"pablo.get_center(i)\".
            self.centers[i, :] = self.pablo.get_center(i)[:2]

        self.comm_dictionary.update({"world communicator" : self.comm})
        self.comm_dictionary.update({"communicator" : self.comm})
        self.comm_dictionary.update({"octree" : self.pablo})
    
    # Testing number of processes inside the communicator.
    def test_comm_size(self):
        """Method which controls that the number of processes of the 
           attributes \"self.comm\" is the same of the one returned by
           the attribute \"comm\" of the class \"ExactSolution2D\". """

        exact_solution = ExactSolution2D.ExactSolution2D(self.comm_dictionary)

        n_procs = self.comm.Get_size()
        # For the method \"assertEqual\", REMEMBER to add first the \"self\", 
        # because it is a method inherited from class \"unittest.TestCase\".
        # http://stackoverflow.com/questions/17779526/python-nameerror-global-name-assertequal-is-not-defined
        self.assertEqual(exact_solution.comm.Get_size(), n_procs)

    # Testing right behaviour of the code if the \"world communicator\" key is
    # None.
    def test_comm_w_null(self):
        """Method which controls that, being the \"world communicator\" None,
           the program launches a \"sys.exit(1)\" and that on the log file, 
           this behaviour will be reported."""

        self.comm_dictionary["world communicator"] = None
        # Check sys exit.
        check_s_e = False
        
        try:
            exact_solution = \
                        ExactSolution2D.ExactSolution2D(self.comm_dictionary)
        # \"sys.exit()\" produce a \"SystemExit\" exception, so we can catch it.
        except SystemExit:
            check_s_e = True
        finally:
            self.assertEqual(check_s_e, True)

    # If you want to see the \"MPI Abort\", comment the following decorator.
    # We used it because we knew the test would have exited before finishing.
    @unittest.skip("Being the world communicator different from None, \"MPI "\
                   "Abort\" is called.")
    def test_comm_null(self):
        self.comm_dictionary["communicator"] = None
        # Check comm null.
        check_c_n = False

        exact_solution = ExactSolution2D.ExactSolution2D(self.comm_dictionary)
        # The \"with\" statement prevent file to not remain opened (it closes
        # them).
        with open(log_file, "r") as of:
            for line in of:
                if "\"MPI Abort\" called during initialization " in line:
                    check_c_n = True

        self.assertEqual(check_c_n, True)

    # Deriving from \"unittest.TestCase\", this class can overload this 
    # method to clear the environment after each test contained in the class.
    def tearDown(self):
        del self.pablo
        del self.centers
        del self.comm
        del self.comm_dictionary

if __name__ == "__main__":
    # http://stackoverflow.com/questions/82831/check-whether-a-file-exists-using-python
    if os.path.exists(log_file):
        with open(log_file, "w") as of:
            pass
    #unittest.main(exit = False)
    suite = unittest.TestLoader().loadTestsFromTestCase(ExactSolution2DTest)
    unittest.TextTestRunner().run(suite)
