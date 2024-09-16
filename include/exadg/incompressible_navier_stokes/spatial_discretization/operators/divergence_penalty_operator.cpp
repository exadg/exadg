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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/divergence_penalty_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
DivergencePenaltyOperator<dim, Number>::DivergencePenaltyOperator() : matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  DivergencePenaltyData const &           data,
  std::shared_ptr<Kernel> const           kernel)
{
  this->matrix_free = &matrix_free;
  this->data        = data;
  this->kernel      = kernel;
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::update(VectorType const & velocity)
{
  kernel->calculate_penalty_parameter(velocity);
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, false);
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    kernel->reinit_cell(integrator);

    do_cell_integral(integrator);

    integrator.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_divergence(kernel->get_volume_flux(integrator, q), q);
  }
}

template class DivergencePenaltyOperator<2, float>;
template class DivergencePenaltyOperator<2, double>;

template class DivergencePenaltyOperator<3, float>;
template class DivergencePenaltyOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
