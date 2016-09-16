// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#include "MyPabloUniform.hpp"

namespace bitpit {
// =================================================================================== //
// NAME SPACES                                                                         //
// =================================================================================== //
using namespace std;

// =================================================================================== //
// CLASS IMPLEMENTATION                                                                    //
// =================================================================================== //

// =================================================================================== //
// CONSTRUCTORS AND OPERATORS
// =================================================================================== //
#if BITPIT_ENABLE_MPI==1
MyPabloUniform::MyPabloUniform(uint8_t dim, int8_t maxlevel, std::string logfile, MPI_Comm comm):ParaTree(dim,maxlevel,logfile,comm){
	m_origin = { {0,0,0} };
	m_L = 1.0;
};

MyPabloUniform::MyPabloUniform(double X, double Y, double Z, double L, uint8_t dim, int8_t maxlevel, std::string logfile, MPI_Comm comm):ParaTree(dim,maxlevel,logfile,comm){
	m_origin[0] = X;
	m_origin[1] = Y;
	m_origin[2] = Z;
	m_L = L;
};
#else
MyPabloUniform::MyPabloUniform(uint8_t dim, int8_t maxlevel, std::string logfile):ParaTree(dim,maxlevel,logfile){
	m_origin = { {0,0,0} };
	m_L = 1.0;
};

MyPabloUniform::MyPabloUniform(double X, double Y, double Z, double L, uint8_t dim, int8_t maxlevel, std::string logfile):ParaTree(dim,maxlevel,logfile){
	m_origin[0] = X;
	m_origin[1] = Y;
	m_origin[2] = Z;
	m_L = L;
};
#endif

// =================================================================================== //
// METHODS
// =================================================================================== //
const u32vector2D& 
MyPabloUniform::_getConnectivity(){
	return ParaTree::getConnectivity();
};

const u32vector2D& 
MyPabloUniform::_getGhostConnectivity(){
	return ParaTree::getConnectivity();
};

uint32_t 
MyPabloUniform::_getNumGhosts() const{
	return ParaTree::getNumGhosts();
};

uint32_t 
MyPabloUniform::_getNumNodes() const{
	return ParaTree::getNumNodes();
};

uint8_t 
MyPabloUniform::_getNnodes(){
	return ParaTree::getNnodes();
};

//darray3 
//MyPabloUniform::_getGhostNodeCoordinates(uint32_t inode){
//	darray3 nodes, nodes_ = ParaTree::getGhostNodeCoordinates(inode);
//	for (int i=0; i<3; i++){
//		nodes[i] = m_origin[i] + m_L * nodes_[i];
//	}
//	return nodes;
//};

darray3 
MyPabloUniform::_getNodeCoordinates(uint32_t inode){
	darray3 nodes, nodes_ = ParaTree::getNodeCoordinates(inode);
	for (int i=0; i<3; i++){
		nodes[i] = m_origin[i] + m_L * nodes_[i];
	}
	return nodes;
};
	
//const u32arr3vector& 
//MyPabloUniform::_getGhostNodes(){
//	return ParaTree::getGhostNodes();
//};

int
MyPabloUniform::_getRank(){
	return ParaTree::getRank();
};

int
MyPabloUniform::_getNproc(){
	return ParaTree::getNproc();
};

// =================================================================================== //
// BASIC GET/SET METHODS															   //
// =================================================================================== //
darray3
MyPabloUniform::_getOrigin(){
	return m_origin;
};

double
MyPabloUniform::_getX0(){
	return m_origin[0];
};

double
MyPabloUniform::_getY0(){
	return m_origin[1];
};

double
MyPabloUniform::_getZ0(){
	return m_origin[2];
};

double
MyPabloUniform::_getL(){
	return m_L;
};

/*! Set the length of the domain.
 * \param[in] Length of the octree.
 */
void
MyPabloUniform::_setL(double L){
	m_L = L;
};

/*! Set the origin of the domain.
 * \param[in] Oriin of the octree.
 */
void
MyPabloUniform::_setOrigin(darray3 origin){
	m_origin = origin;
};

/*! Get the size of an octant corresponding to a target level.
 * \param[in] idx Input level.
 * \return Size of an octant of input level.
 */
double
MyPabloUniform::_levelToSize(uint8_t & level) {
	double size = ParaTree::levelToSize(level);
	return m_L *size;
}

// =================================================================================== //
// INDEX BASED METHODS																   //
// =================================================================================== //
darray3
MyPabloUniform::_getCoordinates(uint32_t idx){
	darray3 coords, coords_;
	coords_ = ParaTree::getCoordinates(idx);
	for (int i=0; i<3; i++){
		coords[i] = m_origin[i] + m_L * coords_[i];
	}
	return coords;
};

double
MyPabloUniform::_getX(uint32_t idx){
	double X, X_;
	X_ = ParaTree::getX(idx);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getY(uint32_t idx){
	double X, X_;
	X_ = ParaTree::getY(idx);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getZ(uint32_t idx){
	double X, X_;
	X_ = ParaTree::getZ(idx);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getSize(uint32_t idx){
	return m_L * ParaTree::getSize(idx);
};

double
MyPabloUniform::_getArea(uint32_t idx){
	return m_L * m_L * ParaTree::getArea(idx);
};

double
MyPabloUniform::_getVolume(uint32_t idx){
	return m_L * m_L * m_L* ParaTree::getArea(idx);
};

void
MyPabloUniform::_getCenter(uint32_t idx, darray3& center){
	darray3 center_ = ParaTree::getCenter(idx);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
};

darray3
MyPabloUniform::_getCenter(uint32_t idx){
	darray3 center, center_ = ParaTree::getCenter(idx);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
	return center;
};

void
MyPabloUniform::_getFaceCenter(uint32_t idx, uint8_t iface, darray3& center){
	darray3 center_ = ParaTree::getFaceCenter(idx, iface);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
};

darray3
MyPabloUniform::_getFaceCenter(uint32_t idx, uint8_t iface){
	darray3 center, center_ = ParaTree::getFaceCenter(idx, iface);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
	return center;
};

darray3
MyPabloUniform::_getNode(uint32_t idx, uint8_t inode){
	darray3 node, node_ = ParaTree::getNode(idx, inode);
	for (int i=0; i<3; i++){
		node[i] = m_origin[i] + m_L * node_[i];
	}
	return node;
};

void
MyPabloUniform::_getNode(uint32_t idx, uint8_t inode, darray3& node){
	darray3 node_ = ParaTree::getNode(idx, inode);
	for (int i=0; i<3; i++){
		node[i] = m_origin[i] + m_L * node_[i];
	}
};

void
MyPabloUniform::_getNodes(uint32_t idx, darr3vector & nodes){
	darray3vector nodes_ = ParaTree::getNodes(idx);
	nodes.resize(ParaTree::getNnodes());
	for (int j=0; j<ParaTree::getNnodes(); j++){
		for (int i=0; i<3; i++){
			nodes[j][i] = m_origin[i] + m_L * nodes_[j][i];
		}
	}
};

darr3vector
MyPabloUniform::_getNodes(uint32_t idx){
	darray3vector nodes, nodes_ = ParaTree::getNodes(idx);
	nodes.resize(ParaTree::getNnodes());
	for (int j=0; j<ParaTree::getNnodes(); j++){
		for (int i=0; i<3; i++){
			nodes[j][i] = m_origin[i] + m_L * nodes_[j][i];
		}
	}
	return nodes;
};

void
MyPabloUniform::_getNormal(uint32_t idx, uint8_t & iface, darray3 & normal) {
	ParaTree::getNormal(idx, iface, normal);
}

darray3
MyPabloUniform::_getNormal(uint32_t idx, uint8_t & iface){
	return ParaTree::getNormal(idx, iface);
}

// =================================================================================== //
// POINTER BASED METHODS															   //
// =================================================================================== //
darray3
MyPabloUniform::_getCoordinates(Octant* oct){
	darray3 coords, coords_;
	coords_ = ParaTree::getCoordinates(oct);
	for (int i=0; i<3; i++){
		coords[i] = m_origin[i] + m_L * coords_[i];
	}
	return coords;
};

double
MyPabloUniform::_getX(Octant* oct){
	double X, X_;
	X_ = ParaTree::getX(oct);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getY(Octant* oct){
	double X, X_;
	X_ = ParaTree::getY(oct);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getZ(Octant* oct){
	double X, X_;
	X_ = ParaTree::getZ(oct);
	X = m_origin[0] + m_L * X_;
	return X;
};

double
MyPabloUniform::_getSize(Octant* oct){
	return m_L * ParaTree::getSize(oct);
};

double
MyPabloUniform::_getArea(Octant* oct){
	return m_L * m_L * ParaTree::getArea(oct);
};

double
MyPabloUniform::_getVolume(Octant* oct){
	return m_L * m_L * m_L* ParaTree::getArea(oct);
};

void
MyPabloUniform::_getCenter(Octant* oct, darray3& center){
	darray3 center_ = ParaTree::getCenter(oct);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
};

darray3
MyPabloUniform::_getCenter(Octant* oct){
	darray3 center, center_ = ParaTree::getCenter(oct);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
	return center;
};

void
MyPabloUniform::_getFaceCenter(Octant* oct, uint8_t iface, darray3& center){
	darray3 center_ = ParaTree::getFaceCenter(oct, iface);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
};

darray3
MyPabloUniform::_getFaceCenter(Octant* oct, uint8_t iface){
	darray3 center, center_ = ParaTree::getFaceCenter(oct, iface);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center_[i];
	}
	return center;
};

darray3
MyPabloUniform::_getNode(Octant* oct, uint8_t inode){
	darray3 node, node_ = ParaTree::getNode(oct, inode);
	for (int i=0; i<3; i++){
		node[i] = m_origin[i] + m_L * node_[i];
	}
	return node;
};

void
MyPabloUniform::_getNode(Octant* oct, uint8_t inode, darray3& node){
	darray3 node_ = ParaTree::getNode(oct, inode);
	for (int i=0; i<3; i++){
		node[i] = m_origin[i] + m_L * node_[i];
	}
};

void
MyPabloUniform::_getNodes(Octant* oct, darr3vector & nodes){
	darray3vector nodes_ = ParaTree::getNodes(oct);
	nodes.resize(ParaTree::getNnodes());
	for (int j=0; j<ParaTree::getNnodes(); j++){
		for (int i=0; i<3; i++){
			nodes[j][i] = m_origin[i] + m_L * nodes_[j][i];
		}
	}
};

darr3vector
MyPabloUniform::_getNodes(Octant* oct){
	darray3vector nodes, nodes_ = ParaTree::getNodes(oct);
	nodes.resize(ParaTree::getNnodes());
	for (int j=0; j<ParaTree::getNnodes(); j++){
		for (int i=0; i<3; i++){
			nodes[j][i] = m_origin[i] + m_L * nodes_[j][i];
		}
	}
	return nodes;
};

void
MyPabloUniform::_getNormal(Octant* oct, uint8_t & iface, darray3 & normal) {
	ParaTree::getNormal(oct, iface, normal);
}

darray3
MyPabloUniform::_getNormal(Octant* oct, uint8_t & iface){
	return ParaTree::getNormal(oct, iface);
}

// =================================================================================== //
// LOCAL TREE GET/SET METHODS														   //
// =================================================================================== //
double
MyPabloUniform::_getLocalMaxSize(){
	return m_L * ParaTree::getLocalMaxSize();
};

double
MyPabloUniform::_getLocalMinSize(){
	return m_L * ParaTree::getLocalMinSize();
};


/*! Get the coordinates of the extreme points of a bounding box containing the local tree
 *  \param[out] P0 Array with coordinates of the first point (lowest coordinates);
 *  \param[out] P1 Array with coordinates of the last point (highest coordinates).
 */
void
MyPabloUniform::_getBoundingBox(darray3 & P0, darray3 & P1){
	darray3		cnode, cnode0, cnode1;
	uint32_t 	nocts = ParaTree::getNumOctants();
	uint32_t	id = 0;
	uint8_t 	nnodes = ParaTree::getNnodes();
	cnode0 = ParaTree::getNode(id, 0);
	id = nocts-1;
	cnode1 = ParaTree::getNode(id, nnodes-1);
	copy(begin(P0), end(P0), begin(cnode0));
	copy(begin(P1), end(P1), begin(cnode1));
	for (id=0; id<nocts; id++){
		cnode0 = ParaTree::getNode(id, 0);
		cnode1 = ParaTree::getNode(id, nnodes-1);
		for (int i=0; i<3; i++){
			P0[i] = min(P0[i], (double)cnode0[i]);
			P1[i] = max(P1[i], (double)cnode1[i]);
		}
	}
	for (int i=0; i<3; i++){
		P0[i] = m_origin[i] + m_L * P0[i];
		P1[i] = m_origin[i] + m_L * P1[i];
	}
};


// =================================================================================== //
// INTERSECTION GET/SET METHODS														   //
// =================================================================================== //
double
MyPabloUniform::_getSize(Intersection* inter){
	return m_L * ParaTree::getSize(inter);
};

double
MyPabloUniform::_getArea(Intersection* inter){
	return m_L * m_L * ParaTree::getArea(inter);
};

darray3
MyPabloUniform::_getCenter(Intersection* inter){
	darray3 center = ParaTree::getCenter(inter);
	for (int i=0; i<3; i++){
		center[i] = m_origin[i] + m_L * center[i];
	}
	return center;
}

// =================================================================================== //
// OTHER OCTANT BASED METHODS												    	   //
// =================================================================================== //
Octant* MyPabloUniform::_getPointOwner(darray3 & point){
	for (int i=0; i<3; i++){
		point[i] = (point[i] - m_origin[i])/m_L;
	}
	return ParaTree::getPointOwner(point);
};

uint32_t MyPabloUniform::_getPointOwnerIdx(darray3 & point){
	for (int i=0; i<3; i++){
		point[i] = (point[i] - m_origin[i])/m_L;
	}
	return ParaTree::getPointOwnerIdx(point);
};

}
