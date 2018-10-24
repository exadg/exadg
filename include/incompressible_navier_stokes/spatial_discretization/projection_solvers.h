/*
 * projection_solvers.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_


namespace IncNS
{
/*
 *  Projection solver for projection with divergence penalty term only using a direct solution
 * approach (note that the system of equations is block-diagonal in this case).
 */
template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type,
         typename Operator>
class DirectProjectionSolverDivergencePenalty
  : public IterativeSolverBase<LinearAlgebra::distributed::Vector<value_type>>
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef DirectProjectionSolverDivergencePenalty<dim,
                                                  fe_degree,
                                                  fe_degree_p,
                                                  fe_degree_xwall,
                                                  xwall_quad_rule,
                                                  value_type,
                                                  Operator>
    THIS;

  static const bool         is_xwall = (xwall_quad_rule > 1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear =
    (is_xwall) ? xwall_quad_rule : fe_degree + 1;

  typedef FEEvaluationWrapper<dim,
                              fe_degree,
                              fe_degree_xwall,
                              n_actual_q_points_vel_linear,
                              dim,
                              value_type,
                              is_xwall>
    FEEval_Velocity_Velocity_linear;

  DirectProjectionSolverDivergencePenalty(Operator const & operator_in) : op(operator_in)
  {
  }

  unsigned int
  solve(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    op.get_data().cell_loop(&THIS::local_solve, this, dst, src);

    return 0;
  }

  void
  local_solve(MatrixFree<dim, value_type> const &           data,
              VectorType &                                  dst,
              VectorType const &                            src,
              std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data, op.get_fe_param(), op.get_dof_index());

    std::vector<LAPACKFullMatrix<value_type>> matrices(
      VectorizedArray<value_type>::n_array_elements);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      const unsigned int total_dofs_per_cell = fe_eval_velocity.dofs_per_cell;

      for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

      // div-div penalty parameter: multiply penalty parameter by the time step size
      const VectorizedArray<value_type> tau =
        op.get_time_step_size() * op.get_array_div_penalty_parameter()[cell];

      for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < total_dofs_per_cell; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval_velocity.evaluate(true, true, false);
        for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
        {
          const VectorizedArray<value_type> tau_times_div =
            tau * fe_eval_velocity.get_divergence(q);
          Tensor<2, dim, VectorizedArray<value_type>> test;
          for(unsigned int d = 0; d < dim; ++d)
            test[d][d] = tau_times_div;
          fe_eval_velocity.submit_gradient(test, q);
          fe_eval_velocity.submit_value(fe_eval_velocity.get_value(q), q);
        }
        fe_eval_velocity.integrate(true, true);

        for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          if(fe_eval_velocity.component_enriched(v))
          {
            for(unsigned int i = 0; i < total_dofs_per_cell; ++i)
              (matrices[v])(i, j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
          }
          else // this is a non-enriched element
          {
            if(j < fe_eval_velocity.std_dofs_per_cell)
              for(unsigned int i = 0; i < fe_eval_velocity.std_dofs_per_cell; ++i)
                (matrices[v])(i, j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
            else // diagonal
              (matrices[v])(j, j) = 1.0;
          }
        }
      }

      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);

      // compute LU factorization and apply inverse matrices
      for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).compute_lu_factorization();
        Vector<value_type> vector_input(total_dofs_per_cell);
        for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
          vector_input(j) = (fe_eval_velocity.read_cellwise_dof_value(j))[v];

        (matrices[v]).solve(vector_input, false);
        for(unsigned int j = 0; j < total_dofs_per_cell; ++j)
          fe_eval_velocity.write_cellwise_dof_value(j, vector_input(j), v);
      }
      fe_eval_velocity.set_dof_values(dst);
    }
  }

  Operator const & op;
};


} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_SOLVERS_H_ */
