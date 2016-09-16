# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# ------------------------------------IMPORT------------------------------------
import sys
import BaseClass2D
import numpy
import utilities
import copy
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class ExactSolution2D(BaseClass2D.BaseClass2D):
    # http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html
    """Class which evaluates the given exact solution on the coordinates
       (we are in 2 dimensions) passed in. 
       
       Attributes:
           _comm (MPI.Intracomm) : intracommunicator which identify the
                                    process where evaluate the exact solution.
           _octree (class_para_tree.Py_Class_Para_Tree_D2) : PABLO's ParaTree.
           _comm_w (MPI.Intracomm) : global intracommunicator.
           _sol (numpy array) : evaluated solution.
           _s_der (numpy array) : evaluated second derivative."""

    # Exact solution = sin((x - 0.5)^2 + (y - 0.5)^2).
    # http://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
    @staticmethod
    def solution(x      , 
                 y      ,
                 z      ,
                 mapping,
                 use_mapping = True):
        """Static method which returns the solution:  
           sin((x - 0.5)^2 + (y - 0.5)^2).
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.
               
           Returns:
               a numpy vector with the solution."""

        dim = 2
        log_file = None
        logger = None
        x_n, y_n = ([] for i in xrange(0, 2))
        x_t = x
        y_t = y
        if (use_mapping):
            if (type(x) is not list):
                x_t = [x]
                y_t = [y]
            n_x = len(x_t)
            for i in xrange(0, n_x):
                values = utilities.apply_persp_trans(dim             , 
                                                     (x_t[i], y_t[i]), 
                                                     mapping         ,
                                                     logger          ,
                                                     log_file)
                x_n.append(values[0])
                y_n.append(values[1])
            
            x_t = x_n
            y_t = y_n

        return numpy.sin(numpy.power(numpy.array(x_t) - 0.5, 2) + 
                     numpy.power(numpy.array(y_t) - 0.5, 2))

    # Solution second derivative = 4 * cos((x - 0.5)^2 + (y - 0.5)^2) - 
    #                              4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
    #                              ((x - 0.5)^2 + (y - 0.5)^2).
    @staticmethod
    def s_s_der(x,
                y,
                z = None):
        """Static method which returns the solution second derivative: 
           4 * cos((x - 0.5)^2 + (y - 0.5)^2) - 
           4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
           ((x - 0.5)^2 + (y - 0.5)^2).
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.
           
           Returns:
               a numpy vector with the solution second derivative."""

        return (numpy.multiply(numpy.cos(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)),
                               4)                                      -
                numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)), 
                               numpy.multiply(numpy.power(x - 0.5, 2)  + 
                                              numpy.power(y - 0.5, 2),
                                              4)))

    # Solution second derivative x.
    @staticmethod
    def s_s_der_x(x,
                  y,
                  z = None):

        return (numpy.multiply(numpy.cos(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)),
                               2)                                      -
                numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)), 
                               numpy.multiply(numpy.power(x - 0.5, 2),
                                              4)))
    # Solution second derivative y.
    @staticmethod
    def s_s_der_y(x,
                  y,
                  z = None):

        return (numpy.multiply(numpy.cos(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)),
                               2)                                      -
                numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)), 
                               numpy.multiply(numpy.power(y - 0.5, 2),
                                              4)))
    @staticmethod
    def multiply_exact_array(array, value):
        return numpy.multiply(array, value)
        
    
    # Solution mixed second derivative = - 4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
    #                                    ((x - 0.5) * (y - 0.5)).
    @staticmethod
    def s_m_s_der(x,
                  y,
                  z = None):
        """Static method which returns the solution mixed second derivative: 
           -4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
           ((x - 0.5) * (y - 0.5)).
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.
           
           Returns:
               a numpy vector with the solution second derivative."""

        return (numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)), 
                               numpy.multiply((x - 0.5), 
                                              numpy.multiply((y - 0.5),
                                                             -4))))
    
    # Evaluate solution.
    def e_sol(self   , 
              x      , 
              y      ,
              z      ,
              mapping,
              use_mapping = True):
        """Method which evaluates the solution. It calls the method \"solution\"
           to obtain it; at a first glance can appear useless, but it has the
           capability to be independent from the exact solution, so for the
           user is useful; one has to change only the method \"solution\" to 
           mantain independence.
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.

           Raises:
               AssertionError : if \"x\" has length different from the one of
                                \"y\" then an AssertionError is Raised and
                                catched, launching an MPI Abort on the 
                                \"_comm_w\" attribute.

           Returns:
               a numpy vector with the solution."""

        # \"Try-catch\" block to assert that vectors \"x\" and \"y\" have the
        # same size.
        try:
            assert len(x) == len(y)
            sol = ExactSolution2D.solution(x      , 
                                           y      ,
                                           z      ,
                                           mapping,
                                           use_mapping)
            msg = "Evaluated exact solution "
            extra_msg = "\":\n" + str(sol)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
            self._sol = sol
        except AssertionError:
            msg = "\"MPI Abort\" called setting exact solution for "
            self.log_msg(msg,
                         "error")
            self._comm_w.Abort(1)

    # Evaluate second derivative.
    def e_s_der(self, 
                x   ,
                y   ,
                z = None):
        """Method which evaluates the second derivative. It calls the method
           \"s_s_der\" to obtain it; at a first glance can appear useless, but
           it has the capability to be independent from the second derivative, so
           for the user is useful; one has to change only the method \"s_s_der\"
           to mantain independence.
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.

           Raises:
               AssertionError : if \"x\" has length different from the one of 
                                \"y\" then an AssertionError is Raised and
                                catched, launching an MPI Abort on the 
                                \"_comm_w\" attribute.

           Returns:
               a numpy vector with the solution second derivative."""
        try:
            assert len(x) == len(y)
            # Second derivative.
            s_der = ExactSolution2D.s_s_der(x, 
                                            y,
                                            z)
            msg = "Evaluated second derivative "
            extra_msg = "\":\n" + str(s_der)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
            self._s_der = s_der
        except AssertionError:
            msg = "\"MPI Abort\" called setting second derivative for "
            self.log_msg(msg)
            self._comm_w.Abort(1)
    
    # Evaluate mixed second derivative.
    def e_m_s_der(self, 
                  x   ,
                  y   ,
                  z = None):
        """Method which evaluates the mixed second derivative. It calls the method
           \"s_s_der\" to obtain it; at a first glance can appear useless, but
           it has the capability to be independent from the second derivative, so
           for the user is useful; one has to change only the method \"s_s_der\"
           to mantain independence.
           
           Arguments:
               x (number or list[numbers]) : abscissa coordinate.
               y (number or list[numbers]) : ordinate coordinate.

           Raises:
               AssertionError : if \"x\" has length different from the one of 
                                \"y\" then an AssertionError is Raised and
                                catched, launching an MPI Abort on the 
                                \"_comm_w\" attribute.

           Returns:
               a numpy vector with the solution second derivative."""
        try:
            assert len(x) == len(y)
            # Mixed second derivative.
            m_s_der = ExactSolution2D.s_m_s_der(x, 
                                                y,
                                                z)
            msg = "Evaluated mixed second derivative "
            extra_msg = "\":\n" + str(m_s_der)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
            self._m_s_der = m_s_der
        except AssertionError:
            msg = "\"MPI Abort\" called setting second derivative for "
            self.log_msg(msg)
            self._comm_w.Abort(1)
    # Evaluate second derivative.
    def e_s_der_x(self, 
                x   ,
                y   ,
                z = None):
        try:
            assert len(x) == len(y)
            # Second derivative.
            s_der_x = ExactSolution2D.s_s_der_x(x, 
                                            y,
                                            z)
            msg = "Evaluated second derivative for x"
            extra_msg = "\":\n" + str(s_der_x)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
            self._s_der_x = s_der_x
        except AssertionError:
            msg = "\"MPI Abort\" called setting second derivative for "
            self.log_msg(msg)
            self._comm_w.Abort(1)
    def e_s_der_y(self, 
                x   ,
                y   ,
                z = None):
        try:
            assert len(x) == len(y)
            # Second derivative.
            s_der_y = ExactSolution2D.s_s_der_y(x, 
                                            y,
                                            z)
            msg = "Evaluated second derivative for y"
            extra_msg = "\":\n" + str(s_der_y)
            self.log_msg(msg   ,
                         "info",
                         extra_msg)
            self._s_der_y = s_der_y
        except AssertionError:
            msg = "\"MPI Abort\" called setting second derivative for "
            self.log_msg(msg)
            self._comm_w.Abort(1)

    @property
    def sol(self):
        return self._sol

    @property
    def s_der(self):
        return self._s_der
    
    @property
    def m_s_der(self):
        return self._m_s_der
    @property
    def s_der_x(self):
        return self._s_der_x
    @property
    def s_der_y(self):
        return self._s_der_y
