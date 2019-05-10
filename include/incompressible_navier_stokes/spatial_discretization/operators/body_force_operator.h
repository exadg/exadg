/*
 * body_force_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

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

template<int dim, typename Number>
class BodyForceOperator
{
public:
  typedef BodyForceOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  BodyForceOperator() : matrix_free(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &    matrix_free_in,
             BodyForceOperatorData<dim> const & operator_data_in)
  {
    this->matrix_free   = &matrix_free_in;
    this->operator_data = operator_data_in;
  }

  void
  evaluate(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

private:
  template<typename CellIntegrator>
  void
  do_cell_integral(CellIntegrator & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      Point<dim, scalar> q_points = integrator.quadrature_point(q);

      vector rhs = evaluate_vectorial_function(operator_data.rhs, q_points, eval_time);

      integrator.submit_value(rhs, q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &,
            Range const & cell_range) const
  {
    Integrator integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      do_cell_integral(integrator);

      integrator.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  BodyForceOperatorData<dim> operator_data;

  mutable Number eval_time;
};
} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_BODY_FORCE_OPERATOR_H_ \
        */
