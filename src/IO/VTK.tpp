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

namespace bitpit{

/*!
 * Adds data strored in std::vector<> to NativeStreamer
 * @tparam T type of std::vector<>
 * @param[in] name name of data set
 * @param[in] data std::vector containing the data
 */
template<class T>
VTKField& VTK::addData( std::string name, std::vector<T> &data ){

    VTKField& field= addData( name, &m_nativeStreamer) ;
    field.setDataType( VTKTypes::whichType(data)) ;

    m_nativeStreamer.addData(name,data) ;

    return field;

}

/*!
 * Add user data for input or output. 
 * Codification will be set according to default value [appended] or to value set by VTK::setDataCodex( VTKFormat ) or VTK::setCodex( VTKFormat )
 * @tparam T type of std::vector<>
 * @param[in] name name of field
 * @param[in] comp type of data field [ VTKFieldType::SCALAR/ VTKFieldType::VECTOR ] 
 * @param[in] loc location of data [VTKLocation::CELL/VTKLocation::POINT]
 * @param[in] data data
 */
template<class T>
VTKField& VTK::addData( std::string name, VTKFieldType comp,  VTKLocation loc, std::vector<T> &data ){

    VTKField&   field = addData( name, data ) ;

    field.setFieldType(comp) ;
    field.setLocation(loc) ;

    return field ;

};

/*!
 * Associates the NativeStreamer to a geometrical field
 * @tparam T type of std::vector<>
 * @param[in] fieldEnum which geometrical field 
 * @param[in] data std::vector containing the data
 */
template<class T>
void VTKUnstructuredGrid::setGeomData( VTKUnstructuredField fieldEnum, std::vector<T> &data ){

    int      index = static_cast<int>(fieldEnum) ;
    VTKField& field = m_geometry[index] ;
    std::string   name = field.getName() ;

    m_nativeStreamer.addData(name,data) ;

    field.setStreamer( m_nativeStreamer) ;
    field.setDataType( VTKTypes::whichType(data)) ;

    return ;
}

/*!
 * Associates the NativeStreamer to a geometrical field
 * @tparam T type of std::vector<>
 * @param[in] fieldEnum which geometrical field 
 * @param[in] data std::vector containing the data
 */
template<class T>
void VTKRectilinearGrid::setGeomData( VTKRectilinearField fieldEnum, std::vector<T> &data ){

    int      index = static_cast<int>(fieldEnum) ;
    VTKField& field = m_geometry[index] ;
    std::string   name = field.getName() ;

    m_nativeStreamer.addData(name,data) ;

    field.setStreamer( m_nativeStreamer) ;
    field.setDataType( VTKTypes::whichType(data)) ;

    return ;
}
}
