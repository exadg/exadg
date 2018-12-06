/*
 * streamfunction_calculator_rhs_operator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "streamfunction_calculator_rhs_operator.h"

namespace IncNS
{
template<int dim, int degree, typename Number>
StreamfunctionCalculatorRHSOperator<dim, degree, Number>::StreamfunctionCalculatorRHSOperator()
  : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
  AssertThrow(dim == 2, ExcMessage("Calculation of streamfunction can only be used for dim==2."));
}

template<int dim, int degree, typename Number>
void
StreamfunctionCalculatorRHSOperator<dim, degree, Number>::initialize(
  MatrixFree<dim, Number> const & data_in,
  unsigned int const              dof_index_u_in,
  unsigned int const              dof_index_u_scalar_in,
  unsigned int const              quad_index_in)
{
  this->data         = &data_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, int degree, typename Number>
void
StreamfunctionCalculatorRHSOperator<dim, degree, Number>::apply(VectorType &       dst,
                                                                VectorType const & src) const
{
  dst = 0;

  data->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, int degree, typename Number>
void
StreamfunctionCalculatorRHSOperator<dim, degree, Number>::cell_loop(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   cell_range) const
{
  FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
  FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.gather_evaluate(src, true, false);

    fe_eval_velocity_scalar.reinit(cell);

    for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
    {
      // we exploit that the (scalar) vorticity is stored in the first component of the vector
      // in case of 2D problems
      fe_eval_velocity_scalar.submit_value(fe_eval_velocity.get_value(q)[0], q);
    }
    fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
  }
}

} // namespace IncNS

#include "streamfunction_calculator_rhs_operator.hpp"
