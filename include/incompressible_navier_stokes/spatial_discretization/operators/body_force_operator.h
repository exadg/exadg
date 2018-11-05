/*
 * body_force_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../functionalities/evaluate_functions.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct BodyForceOperatorData
{
  BodyForceOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;

  std::shared_ptr<Function<dim>> rhs;
};

template<int dim, int degree, typename Number>
class BodyForceOperator
{
public:
  typedef BodyForceOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

  BodyForceOperator() : data(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &    mf_data,
             BodyForceOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  evaluate(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

private:
  template<typename FEEvaluation>
  void
  do_cell_integral(FEEvaluation & fe_eval) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      vector rhs;

      evaluate_vectorial_function(rhs, operator_data.rhs, q_points, eval_time);

      fe_eval.submit_value(rhs, q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &,
            Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  BodyForceOperatorData<dim> operator_data;

  mutable Number eval_time;
};
} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_ \
        */
