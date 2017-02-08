#ifndef MY_CLASS_VTK
#define MY_CLASS_VTK

#include "VTK.hpp"

namespace bitpit {

typedef std::vector<std::vector<uint32_t> > u32vector2D;
typedef std::array<uint32_t, 3> u32array3;
typedef std::array<double, 3> darray3;
typedef std::vector<u32array3> u32arr3vector;
typedef std::vector<darray3> darr3vector;
typedef std::vector<double> dvector; 

using namespace std;
template<class G, class D, int dim>
class My_Class_VTK:
public VTKUnstructuredGrid, public VTKBaseStreamer{
    D* data;
    G& grid;
    const u32vector2D& connectivity;
    //const u32vector2D& ghostConnectivity;
    darr3vector geoNodes;
    darr3vector ghostGeoNodes;

public:
    My_Class_VTK(D* data_    ,
                 G& grid_    ,
                 string dir_ ,
                 string name_,
                 string cod_ ,
                 int ncell_  ,
                 int npoints_,
                 int nconn_) :
    VTKUnstructuredGrid(dir_,
                        name_),
    grid(grid_),
    connectivity(grid_._getConnectivity())
    //ghostConnectivity(grid_.getGhostConnectivity())
    {
        VTKFormat vtk_format;

        if (cod_ == "ascii") {
            vtk_format = VTKFormat::ASCII;    
        }

        this->setCodex(vtk_format);

        this->setDimensions(ncell_  , 
                            npoints_, 
                            nconn_);
                            //VTKElementType::QUAD);
	this->setGeomData(VTKUnstructuredField::POINTS, this);
	this->setGeomData(VTKUnstructuredField::OFFSETS, this);
	this->setGeomData(VTKUnstructuredField::TYPES, this);
	this->setGeomData(VTKUnstructuredField::CONNECTIVITY, this);

        
        data = data_;
        size_t nNodes = grid._getNumNodes();
        //size_t nGhostNodes = grid._getGhostNodes().size();
        
        geoNodes.resize(nNodes);
        //ghostGeoNodes.resize(nGhostNodes);
        
        for (size_t index = 0; index < nNodes; ++index)
                geoNodes[index] = grid._getNodeCoordinates(index);
        
        //for (size_t index = 0; index < nGhostNodes; ++index)
        //        ghostGeoNodes[index] = grid._getGhostNodeCoordinates(index);
    }

    void applyTransf(darr3vector& transGeoNodes) {
        size_t nNodes = grid._getNumNodes();
        //size_t nGhostNodes = grid._getGhostNodes().size();

        for (size_t index = 0; index < nNodes; ++index)
                geoNodes[index] = transGeoNodes[index];
        
        //for (size_t index = 0; index < nGhostNodes; ++index)
        //        ghostGeoNodes[index] = transGhostGeoNodes[index];
    }
        

    void flushData(fstream &str   ,
		   string name    ,
                   VTKFormat codex) {

        int index;
        string vtk_format;
        int nNodes = geoNodes.size();
        //int nGhostNodes = ghostGeoNodes.size();
        int nElements = connectivity.size();
        //int nGhostElements = ghostConnectivity.size();
        int nNodesPerElement = pow(2, dim);

        string indent("         ");

        if (codex == VTKFormat::ASCII) {
            vtk_format = "ascii";
        }

        if (vtk_format == "ascii") {
            if (name == "Points") {
                for (index = 0; index < nNodes; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, 3, geoNodes.at(index));
                    str << endl;
                }
                //for (index = 0; index < nGhostNodes; ++index) {
                //    genericIO::flushASCII(str, indent);
                //    genericIO::flushASCII(str, 3, ghostGeoNodes.at(index));
                //    str << endl;
                //}
            }
            else if (name == "connectivity") {
                for (index = 0; index < nElements; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str,
                                nNodesPerElement,
                                connectivity.at(index));
                    str << endl;
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //    genericIO::flushASCII(str, indent);
                //    vector<uint32_t> gEleConnectivity = ghostConnectivity.at(index);
                //        for (int i = 0; i < nNodesPerElement; ++i)
                //            gEleConnectivity[i] += nNodes;

                //        genericIO::flushASCII(str, nNodesPerElement, gEleConnectivity);

                //        str << endl;
                //}
            }
            else if (name == "types") {
                int type(dim == 2 ? 8 : 11);

                for (index = 0; index < nElements; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, type);
                    str << endl;
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //    genericIO::flushASCII(str, indent);
                //    genericIO::flushASCII(str, type);
                //    str << endl;
                //}
            }
            else if (name == "offsets") {
                int off(0);
                int type(dim == 2 ? 8 : 11);
                int numberOfElements(type == 8 ? 4 : 8);

                for (index = 0; index < nElements; ++index) {
                    off += numberOfElements;
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, off);
                    str << endl;
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //    off += numberOfElements;
                //    genericIO::flushASCII(str, indent);
                //    genericIO::flushASCII(str, off);
                //    str << endl;
                //}
            }
            else if (name == "exact") {
                for (index = 0; index < nElements; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, data[index]);
                    str << endl;
                }
            }
            else if (name == "evaluated") {
                for (index = nElements; index < nElements * 2; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, data[index]);
                    str << endl;
                }
            }
            else if (name == "residual") {
                for (index = nElements * 2; index < nElements * 3; ++index) {
                    genericIO::flushASCII(str, indent);
                    genericIO::flushASCII(str, data[index]);
                    str << endl;
                }
            }
        }
        else {
            if (name == "Points") {
                for (index = 0; index < nNodes; ++index) {
                }
                //for (index = 0; index < nGhostNodes; ++index) {
                //}

            }
            else if (name == "connectivity") {
                for (index = 0; index < nElements; ++index) {
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //}

            }
            else if (name == "types") {
                for (index = 0; index < nElements; ++index) {
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //}
            }
            else if (name == "offset") {
                for (index = 0; index < nElements; ++index) {
                }
                //for (index = 0; index < nGhostElements; ++index) {
                //}
            }
        }
    }
    
    void printVTK() {
	int nproc = grid._getNproc();
	int rank = grid._getRank();
        this->setParallel(nproc, rank);
        this->write();
    }

    void AddData(string name_,
                 int comp_   ,
                 string type_,
                 string loc_ ,
                 string cod_) {
        VTKFieldType field_type;
        VTKDataType data_type;
        VTKFormat format;
        VTKLocation location;

        if (comp_ == 1) {
            field_type = VTKFieldType::SCALAR;
        }
        if (type_ == "Float64") {
            data_type = VTKDataType::Float64;
        }
        if (cod_ == "ascii") {
            format = VTKFormat::ASCII;
        }
        if (loc_ == "Cell") {
            location = VTKLocation::CELL;
        }
    
        this->addData(name_, field_type, location, data_type, this);
    }
};

}

#endif



