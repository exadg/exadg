/*
 * linear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "linear_operator.h"

#include "boundary_conditions.h"
#include "continuum_mechanics.h"

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
    tensor const sigma = material->apply_C(gradient, integrator.get_cell_index(), q);

    // test with gradients
    integrator.submit_gradient(sigma, q);

    if(this->operator_data.unsteady)
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
  }
}

template<int dim, typename Number>
void
LinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &           integrator_m,
  types::boundary_id const & boundary_id) const
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
