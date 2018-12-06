/*
 * divergence_calculator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "divergence_calculator.h"

namespace IncNS
{
template<int dim, int degree, typename Number>
DivergenceCalculator<dim, degree, Number>::DivergenceCalculator()
  : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, int degree, typename Number>
void
DivergenceCalculator<dim, degree, Number>::initialize(MatrixFree<dim, Number> const & data_in,
                                                      unsigned int const dof_index_u_in,
                                                      unsigned int const dof_index_u_scalar_in,
                                                      unsigned int const quad_index_in)
{
  this->data         = &data_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, int degree, typename Number>
void
DivergenceCalculator<dim, degree, Number>::compute_divergence(VectorType &       dst,
                                                              VectorType const & src) const
{
  dst = 0;

  data->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, int degree, typename Number>
void
DivergenceCalculator<dim, degree, Number>::cell_loop(MatrixFree<dim, Number> const & data,
                                                     VectorType &                    dst,
                                                     VectorType const &              src,
                                                     Range const & cell_range) const
{
  FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
  FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.gather_evaluate(src, false, true);

    fe_eval_velocity_scalar.reinit(cell);

    for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
    {
      scalar div = fe_eval_velocity.get_divergence(q);
      fe_eval_velocity_scalar.submit_value(div, q);
    }

    fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
  }
}

} // namespace IncNS

#include "divergence_calculator.hpp"
