/*
 * operator_data.h
 *
 *  Created on: 18.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_DATA_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_DATA_H_

#include "../../../operators/operator_base.h"

#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/enum_types.h"
#include "../../user_interface/material_descriptor.h"

namespace Structure
{
template<int dim>
struct OperatorData : public OperatorBaseData
{
  OperatorData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */), updated_formulation(false)
  {
  }

  std::shared_ptr<BoundaryDescriptor<dim>> bc;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;

  bool updated_formulation;
};

} // namespace Structure

#endif
