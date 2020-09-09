/*
 * divergence_penalty_operator.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/divergence_penalty_operator.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
DivergencePenaltyOperator<dim, Number>::DivergencePenaltyOperator() : matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
DivergencePenaltyOperator<dim, Number>::initialize(MatrixFree<dim, Number> const & matrix_free,
                                                   DivergencePenaltyData const &   data,
                                                   std::shared_ptr<Kernel> const   kernel)
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
DivergencePenaltyOperator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                                  VectorType &                    dst,
                                                  VectorType const &              src,
                                                  Range const &                   range) const
{
  IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.gather_evaluate(src, false, true);

    kernel->reinit_cell(integrator);

    do_cell_integral(integrator);

    integrator.integrate_scatter(false, true, dst);
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
