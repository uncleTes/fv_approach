#ifndef __BITPIT_MY_PABLO_UNIFORM_HPP__
#define __BITPIT_MY_PABLO_UNIFORM_HPP__

// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#include "ParaTree.hpp"
namespace bitpit {
// =================================================================================== //
// TYPEDEFS																			   //
// =================================================================================== //
typedef std::vector<bool>				bvector;
typedef std::bitset<72>					octantID;
typedef std::vector<Octant*>			ptroctvector;
typedef ptroctvector::iterator			octantIterator;
typedef std::vector<darray3>			darray3vector;


// =================================================================================== //
// CLASS DEFINITION                                                                    //
// =================================================================================== //
/*!
 *  \ingroup        PABLO
 *  @{
 *	\date			25/jan/2016
 *	\authors		Edoardo Lombardi
 *	\authors		Marco Cisternino
 *
 *	\brief PABLO Uniform is an example of user class derived from ParaTree to map
 *	ParaTree in a uniform (quadratic/cubic) domain.
 *	Pablo Uniform takes as input in constructor the coordinates of the origin (X,Y,Z) and the length of the side L.
 *
 *	Class MyPabloUniform has a dimensional parameter int dim and it accepts
 *	only two values: dim=2 and dim=3, for 2D and 3D respectively.
 */
class MyPabloUniform : public ParaTree
{
	// =================================================================================== //
	// MEMBERS																			   //
	// =================================================================================== //
private:
	darray3 	m_origin;				/**<Coordinate X,Y,Z of the origin of the octree in the physical domain*/
	double 		m_L;					/**<Side length of octree in the physical domain*/

	// =================================================================================== //
	// CONSTRUCTORS AND OPERATORS
	// =================================================================================== //
public:
#if BITPIT_ENABLE_MPI==1
	MyPabloUniform(uint8_t dim = 2, int8_t maxlevel = 20, std::string logfile="PABLO.log", MPI_Comm comm = MPI_COMM_WORLD);
	MyPabloUniform(double X, double Y, double Z, double L, uint8_t dim = 2, int8_t maxlevel = 20, std::string logfile="PABLO.log", MPI_Comm comm = MPI_COMM_WORLD);
#else
	MyPabloUniform(uint8_t dim = 2, int8_t maxlevel = 20, std::string logfile="PABLO.log");
	MyPabloUniform(double X, double Y, double Z, double L, uint8_t dim = 2, int8_t maxlevel = 20, std::string logfile="PABLO.log");
#endif

	// =================================================================================== //
	// METHODS
	// =================================================================================== //
	const u32vector2D& _getConnectivity();
	
	const u32vector2D& _getGhostConnectivity();
	
	const u32arr3vector& getNodes();
	
	uint32_t _getNumGhosts() const;
	
	uint32_t _getNumNodes() const;
	
	uint8_t _getNnodes();
	
	//darray3 _getGhostNodeCoordinates(uint32_t inode);
	
	darray3 _getNodeCoordinates(uint32_t inode);
	
	//const u32arr3vector & _getGhostNodes();
	
	int _getRank();

	int _getNproc();
	
	// =================================================================================== //
	// BASIC GET/SET METHODS															   //
	// =================================================================================== //
	darray3		_getOrigin();
	double		_getX0();
	double		_getY0();
	double		_getZ0();
	double		_getL();
	void		_setL(double L);
	void		_setOrigin(darray3 origin);
	double		_levelToSize( uint8_t& level);

	// =================================================================================== //
	// INDEX BASED METHODS																   //
	// =================================================================================== //
	darray3 	_getCoordinates(uint32_t idx);
	double 		_getX(uint32_t idx);
	double 		_getY(uint32_t idx);
	double 		_getZ(uint32_t idx);
	double 		_getSize(uint32_t idx);
	double 		_getArea(uint32_t idx);
	double 		_getVolume(uint32_t idx);
	void 		_getCenter(uint32_t idx, darray3& center);
	darray3 	_getCenter(uint32_t idx);
	darray3 	_getFaceCenter(uint32_t idx, uint8_t iface);
	void 		_getFaceCenter(uint32_t idx, uint8_t iface, darray3& center);
	darray3 	_getNode(uint32_t idx, uint8_t inode);
	void 		_getNode(uint32_t idx, uint8_t inode, darray3& node);
	void 		_getNodes(uint32_t idx, darr3vector & nodes);
	darr3vector _getNodes(uint32_t idx);
	void 		_getNormal(uint32_t idx, uint8_t & iface, darray3 & normal);
	darray3 	_getNormal(uint32_t idx, uint8_t & iface);

	// =================================================================================== //
	// POINTER BASED METHODS															   //
	// =================================================================================== //
	darray3 	_getCoordinates(Octant* oct);
	double 		_getX(Octant* oct);
	double 		_getY(Octant* oct);
	double 		_getZ(Octant* oct);
	double 		_getSize(Octant* oct);
	double 		_getArea(Octant* oct);
	double 		_getVolume(Octant* oct);
	void 		_getCenter(Octant* oct, darray3& center);
	darray3 	_getCenter(Octant* oct);
	darray3 	_getFaceCenter(Octant* oct, uint8_t iface);
	void 		_getFaceCenter(Octant* oct, uint8_t iface, darray3& center);
	darray3 	_getNode(Octant* oct, uint8_t inode);
	void 		_getNode(Octant* oct, uint8_t inode, darray3& node);
	void 		_getNodes(Octant* oct, darr3vector & nodes);
	darr3vector _getNodes(Octant* oct);
	void 		_getNormal(Octant* oct, uint8_t & iface, darray3 & normal);
	darray3 	_getNormal(Octant* oct, uint8_t & iface);

	// =================================================================================== //
	// LOCAL TREE GET/SET METHODS														   //
	// =================================================================================== //
	double	 	_getLocalMaxSize();
	double	 	_getLocalMinSize();
	void 		_getBoundingBox(darray3 & P0, darray3 & P1);

	// =================================================================================== //
	// INTERSECTION GET/SET METHODS														   //
	// =================================================================================== //
	double 		_getSize(Intersection* inter);
	double 		_getArea(Intersection* inter);
	darray3 	_getCenter(Intersection* inter);
	darr3vector _getNodes(Intersection* inter);
	darray3 	_getNormal(Intersection* inter);

	// =================================================================================== //
	// OTHER OCTANT BASED METHODS												    	   //
	// =================================================================================== //
	Octant* _getPointOwner(darray3 & point);
	uint32_t _getPointOwnerIdx(darray3 & point);

};

}

#endif /* MYPABLOUNIFORM_HPP_ */
