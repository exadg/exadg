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

#include <exadg/operators/mass_operator.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
MassOperator<dim, n_components, Number>::MassOperator() : scaling_factor(1.0)
{
}

template<int dim, int n_components, typename Number>
void
MassOperator<dim, n_components, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  MassOperatorData<dim> const &             data)
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

// dim components
template class MassOperator<2, 2, float>;
template class MassOperator<2, 2, double>;

template class MassOperator<3, 3, float>;
template class MassOperator<3, 3, double>;

// dim + 1 components
template class MassOperator<2, 3, float>;
template class MassOperator<2, 3, double>;

template class MassOperator<3, 4, float>;
template class MassOperator<3, 4, double>;

} // namespace ExaDG
