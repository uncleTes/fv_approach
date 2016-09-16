/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2016 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of bitbit.
 *
 *  bitpit is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  bitpit is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#include "bitpit_common.hpp"
#include "ParaTree.hpp"

using namespace std;
using namespace bitpit;

// =================================================================================== //
void test003() {

	/**<Instantation and setup of a default (named bitpit) logfile.*/
	int nproc;
	int	rank;
#if BITPIT_ENABLE_MPI==1
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm,&nproc);
	MPI_Comm_rank(comm,&rank);
#else
	nproc = 1;
	rank = 0;
#endif
	log::manager().initialize(log::SEPARATE, false, nproc, rank);
	log::cout() << fileVerbosity(log::NORMAL);
	log::cout() << consoleVerbosity(log::QUIET);

	/**<Instantation of a 2D para_tree object.*/
	ParaTree pablo2;

    /**<Set 2:1 balance only through faces.*/
    pablo2.setBalanceCodimension(1);
    uint32_t idx=0;
    pablo2.setBalance(idx,true);

    /**<Compute the connectivity and write the para_tree.*/
    pablo2.computeConnectivity();
    pablo2.write("Pablo003_iter0");

    /**<Refine globally two level and write the para_tree.*/
    for (int iter=1; iter<3; iter++){
        pablo2.adaptGlobalRefine();
        pablo2.updateConnectivity();
        pablo2.write("Pablo003_iter"+to_string(static_cast<unsigned long long>(iter)));
    }

    /**<Define a center point and a radius.*/
    double xc, yc;
    xc = yc = 0.5;
    double radius = 0.4;

    /**<Simple adapt() [refine] 6 times the octants with at least one node inside the circle.*/
    for (int iter=3; iter<9; iter++){
        uint32_t nocts = pablo2.getNumOctants();
        for (unsigned int i=0; i<nocts; i++){
            /**<Compute the nodes of the octant.*/
            vector<array<double,3> > nodes = pablo2.getNodes(i);
            for (int j=0; j<4; j++){
                double x = nodes[j][0];
                double y = nodes[j][1];
                if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
                    pablo2.setMarker(i, 1);
                }
            }
        }
        /**<Adapt octree.*/
        pablo2.adapt();

        /**<Update the connectivity and write the para_tree.*/
        pablo2.updateConnectivity();
        pablo2.write("Pablo003_iter"+to_string(static_cast<unsigned long long>(iter)));
    }

    /**<Simple adapt() [coarse] 3 times the octants with at least one node inside the 2nd circle.*/
    /**<Define a center point and a radius.*/
    double xc2, yc2;
    xc2 = yc2 = 0.5;
    double radius2 = 0.2;
    for (int iter=9; iter<12; iter++){
        uint32_t nocts = pablo2.getNumOctants();
        for (unsigned int i=0; i<nocts; i++){
            /**<Compute the nodes of the octant.*/
            vector<array<double,3> > nodes = pablo2.getNodes(i);
            for (int j=0; j<4; j++){
                double x = nodes[j][0];
                double y = nodes[j][1];
                if ((pow((x-xc2),2.0)+pow((y-yc2),2.0) <= pow(radius2,2.0))){
                    pablo2.setMarker(i, -1);
                }
            }
        }
        /**<Adapt octree.*/
        pablo2.adapt();

        /**<Update the connectivity and write the para_tree.*/
        pablo2.updateConnectivity();
        pablo2.write("Pablo003_iter"+to_string(static_cast<unsigned long long>(iter)));
    }

    return ;
}


// =================================================================================== //
int main( int argc, char *argv[] ) {

#if BITPIT_ENABLE_MPI==1
	MPI_Init(&argc, &argv);

	{
#else
	BITPIT_UNUSED(argc);
	BITPIT_UNUSED(argv);
#endif
		/**<Calling Pablo Test routines*/
        test003() ;

#if BITPIT_ENABLE_MPI==1
	}

	MPI_Finalize();
#endif
}
