/*
 * projection_solvers.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

namespace IncNS
{
/*
 *  Projection solver for projection with divergence penalty term only using a direct solution
 * approach (note that the system of equations is block-diagonal in this case).
 */
template<int dim, typename Number, typename Operator>
class DirectProjectionSolverDivergencePenalty
  : public IterativeSolverBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DirectProjectionSolverDivergencePenalty<dim, Number, Operator> THIS;

  DirectProjectionSolverDivergencePenalty(Operator const & operator_in) : op(operator_in)
  {
  }

  /*
   * TODO:
   * update_preconditioner is not required here (because the matrix is assembled in this function
   * and because a direct solver is used. This points to deficiencies in the code design, i.e.,
   * this direct solver should not be derived from IterativeSolverBase.
   */
  unsigned int
  solve(VectorType & dst, VectorType const & src, bool const /* update_preconditioner */) const
  {
    dst = 0;

    op.get_data().cell_loop(&THIS::local_solve, this, dst, src);

    return 0;
  }

  void
  local_solve(MatrixFree<dim, Number> const &               data,
              VectorType &                                  dst,
              VectorType const &                            src,
              std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegrator<dim, dim, Number> fe_eval_velocity(data,
                                                      op.get_dof_index(),
                                                      op.get_quad_index());

    std::vector<LAPACKFullMatrix<Number>> matrices(VectorizedArray<Number>::n_array_elements);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      const unsigned int total_dofs_per_cell = fe_eval_velocity.dofs_per_cell;

      for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

      // div-div penalty parameter: multiply penalty parameter by the time step size
      const VectorizedArray<Number> tau =
        op.get_time_step_size() * op.get_array_div_penalty_parameter()[cell];

      for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < total_dofs_per_cell; ++i)
          fe_eval_velocity.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_velocity.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_velocity.evaluate(true, true, false);
        for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
        {
          const VectorizedArray<Number> tau_times_div = tau * fe_eval_velocity.get_divergence(q);
          Tensor<2, dim, VectorizedArray<Number>> test;
          for(unsigned int d = 0; d < dim; ++d)
            test[d][d] = tau_times_div;
          fe_eval_velocity.submit_gradient(test, q);
          fe_eval_velocity.submit_value(fe_eval_velocity.get_value(q), q);
        }
        fe_eval_velocity.integrate(true, true);

        for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          for(unsigned int i = 0; i < total_dofs_per_cell; ++i)
            (matrices[v])(i, j) = fe_eval_velocity.begin_dof_values()[i][v];
        }
      }

      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);

      // compute LU factorization and apply inverse matrices
      for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).compute_lu_factorization();
        Vector<Number> vector_input(total_dofs_per_cell);
        for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
          vector_input(j) = fe_eval_velocity.begin_dof_values()[j][v];

        (matrices[v]).solve(vector_input, false);
        for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
          fe_eval_velocity.begin_dof_values()[j][v] = vector_input(j);
      }
      fe_eval_velocity.set_dof_values(dst);
    }
  }

  Operator const & op;
};


} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_ */
