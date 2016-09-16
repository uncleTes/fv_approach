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
#if BITPIT_ENABLE_MPI==1

#include <mpi.h>
#include <unordered_set>

#include "voloctree.hpp"

using namespace std;

namespace bitpit {

/*!
	\ingroup voloctree
	@{
*/

/*!
	Sets the MPI communicator to be used for parallel communications.

	\param communicator is the communicator to be used for parallel
	communications.
*/
void VolOctree::setCommunicator(MPI_Comm communicator)
{
	PatchKernel::setCommunicator(communicator);

	m_tree.setComm(communicator);
}

/*!
	Updates the partition to optimize the load balance.

	\param trackChanges if set to true the changes to the patch will be
	tracked
	\result Returns all the changes applied to the patch.
*/
const std::vector<adaption::Info> VolOctree::_balancePartition(bool trackChanges)
{
	// Updating the tree
	log::cout() << ">> Load balancing...";

	m_tree.loadBalance();
	m_lastTreeOperation = OP_LOAD_BALANCE;

	log::cout() << " Done" << std::endl;

	// Sync the patch
	return sync(trackChanges);
}

/*!
	@}
*/

}

#endif
