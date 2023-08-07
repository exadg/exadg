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
    // engineering strains (material tensor is symmetric)
    tensor const gradient = integrator.get_gradient(q);

    // Cauchy stresses
    tensor const sigma = material->apply_C(gradient, integrator.get_current_cell_index(), q);

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
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector traction;

    if(boundary_type == BoundaryType::Neumann or boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::inhomogeneous or operator_type == OperatorType::full)
      {
        traction -= calculate_neumann_value<dim, Number>(
          q, integrator_m, boundary_type, boundary_id, this->operator_data.bc, this->time);
      }
    }

    if(boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::homogeneous or operator_type == OperatorType::full)
      {
        auto const it = this->operator_data.bc->robin_k_c_p_param.find(boundary_id);

        if(it != this->operator_data.bc->robin_k_c_p_param.end())
        {
          bool const   normal_projection_displacement = it->second.first[0];
          double const coefficient_displacement       = it->second.second[0];

          if(normal_projection_displacement)
          {
            vector const N = integrator_m.get_normal_vector(q);
            traction += N * (coefficient_displacement * (N * integrator_m.get_value(q)));
          }
          else
          {
            traction += coefficient_displacement * integrator_m.get_value(q);
          }

          if(this->operator_data.unsteady)
          {
            bool const   normal_projection_velocity = it->second.first[1];
            double const coefficient_velocity       = it->second.second[1];

            if(normal_projection_velocity)
            {
              vector const N = integrator_m.get_normal_vector(q);
              traction += N * (coefficient_velocity * this->scaling_factor_mass_boundary *
                               (N * integrator_m.get_value(q)));
            }
            else
            {
              traction += coefficient_velocity * this->scaling_factor_mass_boundary *
                          integrator_m.get_value(q);
            }
          }
        }
      }
    }

    integrator_m.submit_value(traction, q);
  }
}

template class LinearOperator<2, float>;
template class LinearOperator<2, double>;

template class LinearOperator<3, float>;
template class LinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
