FV approach for my PhD Thesis.

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

