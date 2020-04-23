/*
 * rhs_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_

// deal.II
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/mapping_flags.h"

#include "elasticity_operator_base.h"

#include "../../../functions_and_boundary_conditions/evaluate_functions.h"

using namespace dealii;

namespace Structure
{
template<int dim>
struct BodyForceData
{
  BodyForceData() : dof_index(0), quad_index(0), pull_back_body_force(false)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<Function<dim>> function;

  bool pull_back_body_force;
};

template<int dim, typename Number>
class BodyForceOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef BodyForceOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

public:
  /*
   * Constructor.
   */
  BodyForceOperator() : matrix_free(nullptr), time(0.0)
  {
  }

  /*
   * Initialization.
   */
  void
  reinit(MatrixFree<dim, Number> const & matrix_free, BodyForceData<dim> const & data)
  {
    this->matrix_free = &matrix_free;
    this->data        = data;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    // gradients because of potential pull-back of body forces
    flags.cells = update_gradients | update_JxW_values | update_quadrature_points;

    return flags;
  }

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, VectorType const & src, double const time) const
  {
    this->time = time;

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector*/);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      if(data.pull_back_body_force)
      {
        integrator.gather_evaluate(src, false, true, false);
      }

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        auto q_points = integrator.quadrature_point(q);
        auto b        = FunctionEvaluator<1, dim, Number>::value(data.function, q_points, time);

        if(data.pull_back_body_force)
        {
          auto F = get_F<dim, Number>(integrator.get_gradient(q));
          // b_0 = dv/dV * b = det(F) * b
          b *= determinant(F);
        }

        integrator.submit_value(b, q);
      }

      integrator.integrate(true, false);
      integrator.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  BodyForceData<dim> data;

  double mutable time;
};

} // namespace Structure

#endif
