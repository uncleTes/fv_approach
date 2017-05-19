## Discretization of the Laplacian Operator Using Multiple Overlapping Grids
## Using a Pythonistic Approach

### Introduction

Adaptive discretizations are important in many *multiscale* problems, where it is critical to reduce the **computational time** while achieving the same or greater accuracy in particular regions of the computational domain.

*Finite differences* applied on *Cartesian* grids are of course a very simple numerical method for solving differential equations, but do not allow adaptive discretizations, forcing the user to **refine** the computational domain globally. Moreover, when the grid is not cartesian, the discretization of the differential operators in space must take into account the metrics, making *grid transformations* a "bit" annoying to handle.

This problem, imposed by physical domain body-fitted grids, can be crossed more easily using a *finite-volume* approach on **octree meshes**.

### Octree and Programming Language

We have chosen to use in our approach *PABLO*, a C++/MPI library for managing parallel linear octree/quadtree, developed at *OPTIMAD Engineering Srl* and now an integrated part
of *bitpit*, a more complex library always developed by *OPTIMAD Engineering Srl*.

But, to make things more interesting, we have chosen to use as programming language Python, lately used more and more also in the *HPC* fields, of course with the correct foresights that an interpreted language requires.

That's why, under the folder *python_wrapper*, you can find *PABLitO*, a Python API written in Cython for *PABLO* and developed by Federico Tesser with the aim of increasing the number of scientific wrapper written for Python and already present (*MPI4Py*, *NumPy*, *PETSc4py*, and so on), with a fully Python compatible *linear octree*.

For any questions about the prject and *PABLitO*, feel free to contact me.

For any questions about *PABLO* or *bitpit*, contact *OPTIMAD Engineering Srl*.

### Forthcoming Researches

1. Heat equation.

2. 3D extension.

3. Hybrid parallelization using an *MPI + X* approach.

### Notes

Up to date with bitpit 1.2, needed for bug corrections.

bitpit is a C++ library for scientific High Performance Computing.
Within bitpit different modules factorize the typical effort which is needed to derived a real-life application code.
Main efforts are dedicated to handle differnt types of computational meshes, their runtime adaptation and data transfer for parallel applications.

bitpit is developed and maintained by Optimad engineering srl and is being distributed under the GNU LESSER GENERAL PUBLIC LICENSE version 3.

Things to remember:

1. Modules CG and RBF require LAPACK library. If you want to use your own installed version, without rely on the ones installed by Ubuntu, 
   you have to modify the **CMakeLists.txt** setting the *LAPACK_LIBRARIES*, *BLA_STATIC* variables as it is done at lines 414 and 416, 
   and exporting the variable *CPATH*, equal to the path where we can find the include files for the LAPACK library (in my case: 
   :/usr/local/lib/lapack-3.6.1/LAPACKE/include/). 
   After that, add to the variable *target_link_libraries* inside the files **CMakeLists.txt** of the directories */test/CG/* and */test/RBF/* 
   the flag **-lgfortran**.  

2. Add the flag **-fPIC** to the variables *CMAKE_C_FLAGS_RELEASE* and *CMAKE_CXX_FLAGS_RELEASE* (of course if your build mode is **Release**, otherwise
   add this flag at the variables matching your build type), to be able to link the dynamic libraries created by Cython with the bitpit library.

3. The command to compile the files *.pyx*, using the **setyp.py** modified by me, is (of course in my case; the paths can change accordingly):
   python setup.py build_ext --inplace --PABLO-include-path=../src/PABLO/ --IO-include-path=../src/IO/ --mpi-include-path=/usr/local/lib/openmpi-1.10.0/include/ 
   --extensions-source=file_to_compile.pyx  

