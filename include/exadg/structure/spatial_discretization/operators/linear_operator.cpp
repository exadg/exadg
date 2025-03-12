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

#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/linear_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
void
LinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // Cauchy stresses, only valid for linear elasticity
    tensor const sigma =
      material->second_piola_kirchhoff_stress(integrator.get_gradient(q),
                                              integrator.get_current_cell_index(),
                                              q);

    // test with gradients
    integrator.submit_gradient(sigma, q);

    if(this->operator_data.unsteady)
    {
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
    }
  }
}

template<int dim, typename Number>
void
LinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator_m,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    auto const neumann_value = calculate_neumann_value<dim, Number>(
      q, integrator_m, boundary_type, boundary_id, this->operator_data.bc, this->time);

    integrator_m.submit_value(-neumann_value, q);
  }
}

template class LinearOperator<2, float>;
template class LinearOperator<2, double>;

template class LinearOperator<3, float>;
template class LinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
