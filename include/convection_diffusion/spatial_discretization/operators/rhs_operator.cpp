/*
 * rhs_operator.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: fehn
 */

#include "rhs_operator.h"

#include "../../../../include/functionalities/evaluate_functions.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
RHSOperator<dim, degree, Number>::RHSOperator() : data(nullptr), eval_time(0.0)
{
}

template<int dim, int degree, typename Number>
void
RHSOperator<dim, degree, Number>::reinit(MatrixFree<dim, Number> const & mf_data,
                                         RHSOperatorData<dim> const &    operator_data_in)
{
  this->data          = &mf_data;
  this->operator_data = operator_data_in;
}

template<int dim, int degree, typename Number>
void
RHSOperator<dim, degree, Number>::evaluate(VectorType & dst, double const evaluation_time) const
{
  dst = 0;
  evaluate_add(dst, evaluation_time);
}

template<int dim, int degree, typename Number>
void
RHSOperator<dim, degree, Number>::evaluate_add(VectorType & dst, double const evaluation_time) const
{
  this->eval_time = evaluation_time;

  VectorType src;
  data->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, int degree, typename Number>
template<typename FEEvaluation>
void
RHSOperator<dim, degree, Number>::do_cell_integral(FEEvaluation & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

    scalar rhs = evaluate_scalar_function(operator_data.rhs, q_points, eval_time);

    fe_eval.submit_value(rhs, q);
  }
  fe_eval.integrate(true, false);
}

template<int dim, int degree, typename Number>
void
RHSOperator<dim, degree, Number>::cell_loop(MatrixFree<dim, Number> const & data,
                                            VectorType &                    dst,
                                            VectorType const & /*src*/,
                                            Range const & cell_range) const
{
  FEEvaluation<dim, degree, degree + 1, 1, Number> fe_eval(data,
                                                           operator_data.dof_index,
                                                           operator_data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);

    do_cell_integral(fe_eval);

    fe_eval.distribute_local_to_global(dst);
  }
}

} // namespace ConvDiff

#include "rhs_operator.hpp"
