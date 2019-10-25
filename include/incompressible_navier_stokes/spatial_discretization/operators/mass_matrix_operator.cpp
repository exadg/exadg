/*
 * mass_matrix_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "mass_matrix_operator.h"

namespace IncNS
{
template<int dim, int n_components, typename Number>
MassMatrixOperator<dim, n_components, Number>::MassMatrixOperator() : scaling_factor(1.0)
{
}

template<int dim, int n_components, typename Number>
void
MassMatrixOperator<dim, n_components, Number>::reinit(
  MatrixFree<dim, Number> const &   matrix_free,
  AffineConstraints<double> const & constraint_matrix,
  MassMatrixOperatorData const &    data)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  this->integrator_flags = kernel.get_integrator_flags();
}

template<int dim, int n_components, typename Number>
void
MassMatrixOperator<dim, n_components, Number>::set_scaling_factor(Number const & number)
{
  scaling_factor = number;
}

template<int dim, int n_components, typename Number>
void
MassMatrixOperator<dim, n_components, Number>::apply_scale(VectorType &       dst,
                                                           Number const &     factor,
                                                           VectorType const & src) const
{
  scaling_factor = factor;

  this->apply(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
MassMatrixOperator<dim, n_components, Number>::apply_scale_add(VectorType &       dst,
                                                               Number const &     factor,
                                                               VectorType const & src) const
{
  scaling_factor = factor;

  this->apply_add(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
MassMatrixOperator<dim, n_components, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(scaling_factor, integrator.get_value(q)), q);
  }
}

// scalar
template class MassMatrixOperator<2, 1, float>;
template class MassMatrixOperator<2, 1, double>;

template class MassMatrixOperator<3, 1, float>;
template class MassMatrixOperator<3, 1, double>;

// vector
template class MassMatrixOperator<2, 2, float>;
template class MassMatrixOperator<2, 2, double>;

template class MassMatrixOperator<3, 3, float>;
template class MassMatrixOperator<3, 3, double>;

} // namespace IncNS
