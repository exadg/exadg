/*
 * mapping_flags.h
 *
 *  Created on: Jun 12, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_MAPPING_FLAGS_H_
#define INCLUDE_OPERATORS_MAPPING_FLAGS_H_

#include <deal.II/fe/fe_update_flags.h>

namespace ExaDG
{
using namespace dealii;

struct MappingFlags
{
  MappingFlags()
    : cells(update_default), inner_faces(update_default), boundary_faces(update_default)
  {
  }

  MappingFlags
  operator||(MappingFlags const & other)
  {
    MappingFlags flags_combined;

    flags_combined.cells          = this->cells | other.cells;
    flags_combined.inner_faces    = this->inner_faces | other.inner_faces;
    flags_combined.boundary_faces = this->boundary_faces | other.boundary_faces;

    return flags_combined;
  }

  UpdateFlags cells;
  UpdateFlags inner_faces;
  UpdateFlags boundary_faces;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_MAPPING_FLAGS_H_ */
