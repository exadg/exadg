/*
 * mass_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include <exadg/operators/mass_operator.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, int n_components, typename Number>
MassOperator<dim, n_components, Number>::MassOperator() : scaling_factor(1.0)
{
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::initialize(
  MatrixFree<dim, Number> const &   matrix_free,
  AffineConstraints<Number> const & affine_constraints,
  MassOperatorData<dim> const &     data)
{
  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel.get_integrator_flags();
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::set_scaling_factor(Number const & number)
{
  scaling_factor = number;
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::apply_scale(VectorType &       dst,
                                                     Number const &     factor,
                                                     VectorType const & src) const
{
  scaling_factor = factor;

  this->apply(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::apply_scale_add(VectorType &       dst,
                                                         Number const &     factor,
                                                         VectorType const & src) const
{
  scaling_factor = factor;

  this->apply_add(dst, src);

  scaling_factor = 1.0;
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(scaling_factor, integrator.get_value(q)), q);
  }
}

// scalar
template class MassOperator<2, 1, float>;
template class MassOperator<2, 1, double>;

template class MassOperator<3, 1, float>;
template class MassOperator<3, 1, double>;

// vector
template class MassOperator<2, 2, float>;
template class MassOperator<2, 2, double>;

template class MassOperator<3, 3, float>;
template class MassOperator<3, 3, double>;

} // namespace ExaDG
