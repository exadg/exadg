/*
 * mass_matrix_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "mass_matrix_operator.h"

namespace IncNS
{
template<int dim, typename Number>
void
MassMatrixOperator<dim, Number>::set_scaling_factor(Number const & number) const
{
  kernel.set_scaling_factor(number);
}

template<int dim, typename Number>
void
MassMatrixOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free,
                                        AffineConstraints<double> const & constraint_matrix,
                                        MassMatrixOperatorData const &    operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  this->integrator_flags = kernel.get_integrator_flags();
}

template<int dim, typename Number>
void
MassMatrixOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(integrator.get_value(q)), q);
  }
}

template class MassMatrixOperator<2, float>;
template class MassMatrixOperator<2, double>;

template class MassMatrixOperator<3, float>;
template class MassMatrixOperator<3, double>;

} // namespace IncNS
