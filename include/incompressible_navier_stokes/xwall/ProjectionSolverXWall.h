/*
 * ProjectionSolverXWall.h
 *
 *  Created on: 2017 M03 3
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_PROJECTIONSOLVERXWALL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_PROJECTIONSOLVERXWALL_H_


struct ProjectionOperatorData
{
  double penalty_parameter_divergence;
  double penalty_parameter_continuity;
  bool solve_stokes_equations;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorBaseOld: public BaseOperator<dim>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  ProjectionOperatorBaseOld(MatrixFree<dim,value_type> const &data_in,
                            const unsigned int               dof_index_in,
                            const unsigned int               quad_index_in,
                            ProjectionOperatorData const     projection_operator_data_in)
    :
    data(data_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    array_penalty_parameter_divergence(0),
    array_penalty_parameter_continuity(0),
    projection_operator_data(projection_operator_data_in)
  {
    array_penalty_parameter_divergence.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    array_penalty_parameter_continuity.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  }

  void calculate_array_penalty_parameter(parallel::distributed::Vector<value_type> const &velocity_n,
                                         double const                                    cfl,
                                         double const                                    time_step)
  {
    velocity_n.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity_n);
      fe_eval.evaluate (true,false);
      VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
      Tensor<1,dim,VectorizedArray<value_type> > U_mean;
      VectorizedArray<value_type> norm_U_mean;
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        U_mean += JxW_values[q]*fe_eval.get_value(q);
      }
      U_mean /= volume;
      norm_U_mean = U_mean.norm();

      // inverse cfl dependency does not make sense when solving the Stokes equations
      if(projection_operator_data.solve_stokes_equations == true)
      {
//        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim) * time_step;
//        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity * norm_U_mean * time_step;

        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim);
        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity * norm_U_mean;
      }
      else
      {
        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence/cfl * norm_U_mean * std::exp(std::log(volume)/(double)dim) * time_step;
        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity/cfl * norm_U_mean * time_step;
      }
    }
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_divergence() const
  {
    return array_penalty_parameter_divergence;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_continuity() const
  {
    return array_penalty_parameter_continuity;
  }
  unsigned int get_dof_index() const
  {
    return dof_index;
  }
  unsigned int get_quad_index() const
  {
    return quad_index;
  }
  FEParameters<dim> const * get_fe_param() const
  {
    return this->fe_param;
  }

private:
  MatrixFree<dim,value_type> const & data;
  unsigned int const dof_index;
  unsigned int const quad_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_divergence;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_continuity;
  ProjectionOperatorData projection_operator_data;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergencePenaltyXWall : public ProjectionOperatorBaseOld<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  ProjectionOperatorDivergencePenaltyXWall(MatrixFree<dim,value_type> const & data_in,
                                           FEParameters<dim> * fe_param_in,
                                           const unsigned int dof_index_in,
                                           const unsigned int quad_index_in,
                                           ProjectionOperatorData const projection_operator_data_in,
                                           std::shared_ptr< InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> > inv_mass_xw)
    :
    ProjectionOperatorBaseOld<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(data_in, dof_index_in,quad_index_in,projection_operator_data_in),
    fe_eval(1,FEEval_Velocity_Velocity_linear(data_in, fe_param_in, dof_index_in, quad_index_in)),
    inverse_mass_matrix_operator_xwall(inv_mass_xw),
    tau(1),
    curr_cell(0)
  {}

  void setup(const unsigned int cell)
  {
    curr_cell = cell;
    tau[0] = this->get_array_penalty_parameter_divergence()[cell];
    fe_eval[0].reinit(cell);
  }

  void precondition(VectorizedArray<double>       *dst,
                    const VectorizedArray<double> *src) const
  {
    inverse_mass_matrix_operator_xwall->local_apply_inverse_mass_matrix(curr_cell, dst, src);
  }

  void vmult(VectorizedArray<double> *dst,
             VectorizedArray<double> *src) const
  {
    Assert(fe_eval[0].get_shape_info().element_type <=
        dealii::internal::MatrixFreeFunctions::tensor_symmetric,
           ExcNotImplemented());


    // compute matrix vector product on element
    fe_eval[0].evaluate(src, true, true);

    for (unsigned int q=0; q<fe_eval[0].n_q_points; ++q)
    {
      VectorizedArray<double> tau_times_div = tau[0] * fe_eval[0].get_divergence(q);
      Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
        for (unsigned int d=0; d<dim; ++d)
          unit_times_divU[d][d] = tau_times_div;
        fe_eval[0].submit_gradient(unit_times_divU, q);
      fe_eval[0].submit_value (fe_eval[0].get_value(q), q);
    }

    fe_eval[0].integrate(true, true, dst);
  }

private:
  mutable AlignedVector<FEEval_Velocity_Velocity_linear> fe_eval;
  std::shared_ptr< InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> > inverse_mass_matrix_operator_xwall;
  AlignedVector<VectorizedArray<double> > tau;
  unsigned int curr_cell;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule,typename value_type>
class IterativeProjectionSolverDivergencePenaltyXWall: public IterativeProjectionSolverDivergencePenalty<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  IterativeProjectionSolverDivergencePenaltyXWall(ProjectionOperatorBaseOld<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator_in,
                                                  ProjectionSolverData const solver_data_in)
    :
      IterativeProjectionSolverDivergencePenalty<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>(projection_operator_in,solver_data_in)
  {}

  void local_solve(const MatrixFree<dim,value_type>                &data,
                   parallel::distributed::Vector<value_type>       &dst,
                   const parallel::distributed::Vector<value_type> &src,
                   const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    AssertThrow(this->solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix,
      ExcMessage("Specified preconditioner is not available for projection_type = ProjectionType::DivergencePenalty and solver_projection = SolverProjection::PCG"));

    FEEval_Velocity_Velocity_linear fe_eval(data,this->projection_operator->get_fe_param(),this->projection_operator->get_dof_index(),this->projection_operator->get_quad_index());

    ProjectionOperatorDivergencePenaltyXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> *
          projection_operator_div = static_cast<ProjectionOperatorDivergencePenaltyXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * >(this->projection_operator);

    AlignedVector<VectorizedArray<value_type> > source;
    AlignedVector<VectorizedArray<value_type> > solution;
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;
      source.resize(total_dofs_per_cell);
      solution.resize(total_dofs_per_cell);
      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        source[j] = fe_eval.read_cellwise_dof_value(j);

      InternalSolvers::SolverCG<VectorizedArray<double> > cg_solver(total_dofs_per_cell,
                                                                    this->solver_data.solver_tolerance_abs,
                                                                    this->solver_data.solver_tolerance_rel,
                                                                    this->solver_data.max_iter);

      projection_operator_div->setup(cell);

      cg_solver.solve(projection_operator_div, solution.begin(), source.begin());

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j,solution[j]);
      fe_eval.set_dof_values (dst);
    }
  }
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_PROJECTIONSOLVERXWALL_H_ */
