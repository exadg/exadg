/*
 * enum_types.h
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_
#define INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_

#include <string>

/**************************************************************************************/
/*                                                                                    */
/*                                         MESH                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 * Triangulation type
 */
enum class TriangulationType
{
  Undefined,
  Distributed,
  FullyDistributed
};

std::string
enum_to_string(TriangulationType const enum_type);

/*
 *  Mapping type (polynomial degree)
 */
enum class MappingType
{
  Affine,
  Quadratic,
  Cubic,
  Isoparametric
};

std::string
enum_to_string(MappingType const enum_type);


#endif /* INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_ */
