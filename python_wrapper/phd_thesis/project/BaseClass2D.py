# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import utilities
# http://wiki.gwdg.de/index.php/Mpi4py
from mpi4py import MPI
# Module \"sys\" is needed for \"SystemExit\" exceptions. 
import sys
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class BaseClass2D(object):
    # To see python behaviour with default parameters for mutable objects, see
    # http://stackoverflow.com/questions/11381968/unit-testing-objects-in-python-object-is-not-over-written-in-setup
    # http://effbot.org/zone/default-values.htm
    def __init__(self, 
                 kwargs = {}):
        """Initialization method for the \"BaseClass2D\" class.

           Arguments:
               kwargs (dictionary) : it must contains the following keys:
                                     - \"communicator\";
                                     - \"octree\";
                                     - \"world communicator\";
                                     - \"log file\".
                                     
            Raises:
                AssertionError : if one of the attributes \"_comm\", \"_comm_w\"
                                 and  \"_octree\" of the class is None, then the 
                                 exception is raised and catched, launching an 
                                 \"MPI Abort\" on the _comm_w (if is not None, 
                                 otherwise will be called a \"sys.exit(1)\"."""

        comm = kwargs["communicator"]
        octree = kwargs["octree"]
        comm_w = kwargs["world communicator"]
        log_file =  kwargs["log file"]
        to_log = kwargs["to log"]
        what_log = "debug"
        # If \"Log\" option in file \"PABLO.ini\" is \"False\", then the 
        # \"what_log\" flag will be \"critical\", to log only messages with
        # importance equal or greater than \"critical\". But, usually in the
        # context of this project, debug flags are just "info\" and \"error\". 
        if (not to_log):
            what_log = "critical"
        self.logger = utilities.set_class_logger(self    , 
                                                 log_file,
                                                 what_log)
        # Mangling with the prefix \"__\" to have \"private\" attributes. Use 
        # \"_\" instead to have \"protected\" ones.
        # http://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes
        self._comm = utilities.check_mpi_intracomm(comm       , 
                                                   self.logger,
                                                   type = "local")
        self._comm_w = utilities.check_mpi_intracomm(comm_w     ,
                                                     self.logger,
                                                     type = "world")
        self._octree = utilities.check_octree(octree     , 
                                              self._comm , 
                                              self.logger,
                                              type = "local")
        self._dim = kwargs["dimension"]

        initialized = True
        try:
            assert (self._comm is not None), \
                   "No local communicator given "
            assert (self._comm_w is not None), \
                   "No global communicator given "
            assert (self._octree is not None), \
                  "No octree given "
            assert (2 <= self._dim <= 3), \
                   "Wrong dimension chosen for the problem "
            msg = "Initialized class "
        # http://www.tutorialspoint.com/python/assertions_in_python.htm
        # http://stackoverflow.com/questions/20059766/handle-exception-in-init
        except AssertionError:
            msg = sys.exc_info()[1] 
            initialized = False
            if self._comm_w:
                # http://stackoverflow.com/questions/9064079/aborting-execution-of-all-processes-in-mpi
                # http://stackoverflow.com/questions/5433697/terminating-all-processes-with-mpi
                self._comm_w.Abort(1)
            else:
                sys.exit(1)
        # This \"finally\" is executed also after the \"sys.exit()\" because it
        # is not a \"os._exit()\", and so it throws an exception \"SystemExit\".
        # See:
        # http://stackoverflow.com/questions/7709411/why-finally-block-is-executing-after-calling-sys-exit0-in-except-block
        finally:
            self.log_msg(msg,
                         "info" if initialized else "error")
                         
    def __del__(self):
        msg = "Called destructor "
        self.log_msg(msg,
                     "info")

    # Log a message, with an extra message available.
    def log_msg(self ,
                msg  , 
                level,
                extra_msg = ""):
        """Method which write on the logger a message composed by a fixed part
           and a variable one, which is the input argument.
           
           Arguments:
               msg (string) : message to be written with the fixed part.
               level (string) : indicator of the level at which you want to
                                write the msg.
               extra_msg (string, default = "") : extra informations."""

        log_msg = msg                                                    + \
                  " for local comm \""                                   + \
                  str(self._comm.Get_name() if self._comm else None)     + \
                  "\" and world comm \""                                 + \
                  str(self._comm_w.Get_name() if self._comm_w else None) + \
                  "\" and rank \""                                       + \
                  str(self._comm.Get_rank() if self._comm else None)     + \
                  "\" " + extra_msg                                      + \
                  "."
        if level == "info":
            self.logger.info(log_msg)
        elif level == "error":
            self.logger.error(log_msg)

    def init_trans_dict(self,
                        kwargs = {}):
        self._trans_dict = kwargs
    
    def init_trans_adj_dict(self,
                            kwargs = {}):
        self._trans_adj_dict = kwargs

    def get_trans(self,
                  grid):
        return self._trans_dict[grid]
    
    def get_trans_adj(self,
                      grid):
        return self._trans_adj_dict[grid]
    
    # http://stackoverflow.com/questions/15458613/python-why-is-read-only-property-writable
    # https://docs.python.org/2/library/functions.html#property
    # http://stackoverflow.com/questions/6618002/python-property-versus-getters-and-setters
    # http://radek.io/2011/07/21/private-protected-and-public-in-python/
    @property
    def comm(self):
        return self._comm

    @property
    def octree(self):
        return self._octree

    @property
    def comm_w(self):
        return self._comm_w
