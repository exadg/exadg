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

#include <exadg/operators/mass_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct MassOperatorData : public ExaDG::MassOperatorData<dim>
{
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;
};

template<int dim, typename Number>
class MassOperator : public ExaDG::MassOperator<dim, dim, Number>
{
public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MassOperatorData<dim> const &             data)
  {
    operator_data = data;

    ExaDG::MassOperator<dim, dim, Number>::initialize(matrix_free, affine_constraints, data);
  }

private:
  MassOperatorData<dim> operator_data;
};

} // namespace Structure
} // namespace ExaDG



#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_MASS_OPERATOR_H_ */
