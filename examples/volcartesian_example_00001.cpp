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

/*!
	\example volcartesian_example_00001.cpp

	\brief Use volcartesian to create a 2D Cartesian patch

	This example creates a 2D Cartesian patch on the square domain [-10,10]x[-10,10].
	The domain is discretized with a cell size of 0.5 both in x and y directions.

	<b>To run</b>: ./volcartesian_example_00001 \n
*/

#include <array>
#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_volcartesian.hpp"

using namespace bitpit;

int main(int argc, char *argv[]) {

	std::cout << "Creating a 2D Cartesian patch" << "\n";

#if BITPIT_ENABLE_MPI==1
	MPI_Init(&argc,&argv);
#endif

	std::array<double, 3> origin = {{0., 0., 0.}};
	double length = 20;
	double dh = 0.5;

	VolCartesian *patch_2D = new VolCartesian(0, 2, origin, length, dh);
	patch_2D->getVTK().setName("cartesian_2D_patch");
	patch_2D->update();
	patch_2D->write();

#if BITPIT_ENABLE_MPI==1
	MPI_Finalize();
#endif

}
