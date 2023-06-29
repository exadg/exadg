/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_MASS_OPERATOR_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_MASS_OPERATOR_H_

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/operators/mass_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct MassOperatorData : public ExaDG::MassOperatorData<dim>
{
  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};

template<int dim, typename Number>
class MassOperator : public ExaDG::MassOperator<dim, dim, Number>
{
public:
  typedef ExaDG::MassOperator<dim, dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MassOperatorData<dim> const &             data)
  {
    operator_data = data;

    ExaDG::MassOperator<dim, dim, Number>::initialize(matrix_free, affine_constraints, data);
  }

  void
  set_inhomogeneous_boundary_values(VectorType & dst) const final
  {
    std::map<dealii::types::global_dof_index, double> boundary_values;
    for(auto dbc : operator_data.bc->dirichlet_bc_initial_acceleration)
    {
      dbc.second->set_time(this->get_time());
      dealii::ComponentMask mask =
        operator_data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

      dealii::VectorTools::interpolate_boundary_values(
        *this->matrix_free->get_mapping_info().mapping,
        this->matrix_free->get_dof_handler(operator_data.dof_index),
        dbc.first,
        *dbc.second,
        boundary_values,
        mask);
    }

    // set Dirichlet values in solution vector
    for(auto m : boundary_values)
      if(dst.get_partitioner()->in_local_range(m.first))
        dst[m.first] = m.second;

    dst.update_ghost_values();
  }

private:
  MassOperatorData<dim> operator_data;
};

} // namespace Structure
} // namespace ExaDG



#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_MASS_OPERATOR_H_ */
