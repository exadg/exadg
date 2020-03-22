/*
 * mapping_degree.h
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_
#define INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_

#include "enum_types.h"

using namespace dealii;

inline unsigned int
get_mapping_degree(MappingType const & mapping_type, unsigned int const degree_shape_functions)
{
  unsigned int degree = 0;

  switch(mapping_type)
  {
    case MappingType::Affine:
      degree = 1;
      break;
    case MappingType::Quadratic:
      degree = 2;
      break;
    case MappingType::Cubic:
      degree = 3;
      break;
    case MappingType::Isoparametric:
      degree = degree_shape_functions;
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return degree;
}



#endif /* INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_ */
