/*
 * DGNavierStokesCoupled.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESCOUPLED_H_
#define INCLUDE_DGNAVIERSTOKESCOUPLED_H_

using namespace dealii;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;

  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const &triangulation,
                        InputParameters const                           &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter)
  {}

  void setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs);

  // initialization of vectors
  void initialize_block_vector_velocity_pressure(parallel::distributed::BlockVector<value_type> &src) const
  {
    // velocity(1 block) + pressure(1 block)
    src.reinit(2);

    this->data.initialize_dof_vector(src.block(0),
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    this->data.initialize_dof_vector(src.block(1),
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    src.collect_sizes();
  }

  unsigned int solve_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                                         parallel::distributed::BlockVector<value_type> const &src,
                                         parallel::distributed::Vector<value_type> const      *vector_linearization = nullptr);

  unsigned int solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                        parallel::distributed::Vector<value_type> const &sum_alphai_ui);

  void apply_linearized_problem (parallel::distributed::BlockVector<value_type> &dst,
                                 parallel::distributed::BlockVector<value_type> const &src) const;

  void rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                           parallel::distributed::Vector<value_type> const &src) const;

  void evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src,
                                 value_type const                                evaluation_time) const;

  void evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                                    parallel::distributed::BlockVector<value_type> const &src,
                                    parallel::distributed::Vector<value_type> const      &sum_alphai_ui);


private:
  parallel::distributed::BlockVector<value_type> residual;
  parallel::distributed::BlockVector<value_type> increment;
  parallel::distributed::Vector<value_type> temp;
  parallel::distributed::Vector<value_type> velocity_linear;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > /*periodic_face_pairs*/)
{
  if(this->param.solve_stokes_equations == false && this->param.convective_step_implicit == true)
  {
    initialize_block_vector_velocity_pressure(residual);
    initialize_block_vector_velocity_pressure(increment);
    this->initialize_vector_velocity(temp);
    this->initialize_vector_velocity(velocity_linear);
  }

//  // Laplace Operator
//  LaplaceOperatorData<dim> laplace_operator_data;
//  laplace_operator_data.laplace_dof_index = static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure);
//  laplace_operator_data.laplace_quad_index = static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::pressure);
//  laplace_operator_data.penalty_factor = param.IP_factor_pressure;
//
//  // TODO
//  /*
//   * approach of Ferrer et al.: increase penalty parameter when reducing the time step
//   * in order to improve stability in the limit of small time steps
//   */
//
//  /*
//  double dt_ref = 0.1;
//  laplace_operator_data.penalty_factor = param.IP_factor_pressure/time_step*dt_ref;
//  */
//  laplace_operator_data.dirichlet_boundaries = neumann_boundary;
//  laplace_operator_data.neumann_boundaries = dirichlet_boundary;
//  laplace_operator_data.periodic_face_pairs_level0 = periodic_face_pairs;
//  laplace_operator.reinit(data,mapping,laplace_operator_data);
//
//  // Pressure Poisson solver
//  PoissonSolverData poisson_solver_data;
//  poisson_solver_data.solver_tolerance_rel = param.rel_tol_pressure;
//  poisson_solver_data.solver_tolerance_abs = param.abs_tol_pressure;
//  poisson_solver_data.solver_poisson = param.solver_poisson;
//  poisson_solver_data.preconditioner_poisson = param.preconditioner_poisson;
//  poisson_solver_data.multigrid_smoother = param.multigrid_smoother;
//  poisson_solver_data.coarse_solver = param.multigrid_coarse_grid_solver;
//  pressure_poisson_solver.initialize(laplace_operator,mapping,data,poisson_solver_data);

//  // helmholtz operator
//  HelmholtzOperatorData<dim> helmholtz_operator_data;
//  helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity);
//  helmholtz_operator_data.formulation_viscous_term = param.formulation_viscous_term;
//  helmholtz_operator_data.IP_formulation_viscous = param.IP_formulation_viscous;
//  helmholtz_operator_data.IP_factor_viscous = param.IP_factor_viscous;
//  helmholtz_operator_data.dirichlet_boundaries = dirichlet_boundary;
//  helmholtz_operator_data.neumann_boundaries = neumann_boundary;
//  helmholtz_operator_data.viscosity = viscosity;
//  helmholtz_operator_data.mass_matrix_coefficient = gamma0/time_step;
//  helmholtz_operator.reinit(data, mapping, helmholtz_operator_data,fe_param);
//
//  HelmholtzSolverData helmholtz_solver_data;
//  helmholtz_solver_data.solver_viscous = param.solver_viscous;
//  helmholtz_solver_data.preconditioner_viscous = param.preconditioner_viscous;
//  helmholtz_solver_data.solver_tolerance_abs = param.abs_tol_viscous;
//  helmholtz_solver_data.solver_tolerance_rel = param.rel_tol_viscous;
//  helmholtz_solver_data.multigrid_smoother = param.multigrid_smoother_viscous;
//  helmholtz_solver_data.coarse_solver = param.multigrid_coarse_grid_solver_viscous;
//
//  helmholtz_solver.initialize(helmholtz_operator, mapping, data, helmholtz_solver_data,
//      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
//      static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
//      fe_param);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall,typename value_type>
struct NavierStokesCoupledMatrix : public Subscriptor
{
  void initialize(DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
  {
    ns_operation = &ns_op;
  }
  void vmult (parallel::distributed::BlockVector<value_type>        &dst,
              const parallel::distributed::BlockVector<value_type>  &src) const
  {
    ns_operation->apply_linearized_problem(dst,src);
  }
  DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                          parallel::distributed::BlockVector<value_type> const &src) const
{
  // (1,1) block of saddle point matrix
  this->mass_matrix_operator.apply(dst.block(0),src.block(0));
  dst.block(0) *= this->gamma0/this->time_step;

  if(this->param.solve_stokes_equations == false && this->param.convective_step_implicit == true)
    this->convective_operator.apply_linearized_add(dst.block(0),src.block(0),&velocity_linear,this->time+this->time_step);

  this->viscous_operator.apply_add(dst.block(0),src.block(0));

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply_add(dst.block(0),src.block(1));

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1),src.block(0));
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                    parallel::distributed::Vector<value_type> const      &src) const
{
  // velocity-block
  this->mass_matrix_operator.apply(dst.block(0),src);
  this->body_force_operator.evaluate_add(dst.block(0),this->time+this->time_step);
  this->viscous_operator.rhs_add(dst.block(0),this->time+this->time_step);
  this->gradient_operator.rhs_add(dst.block(0),this->time+this->time_step);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1),this->time+this->time_step);
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                          parallel::distributed::Vector<value_type> const &src,
                          value_type const                                evaluation_time) const
{
  this->convective_operator.evaluate(dst,src,evaluation_time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src,
                             parallel::distributed::Vector<value_type> const      &sum_alphai_ui)
{
  // velocity-block
  this->body_force_operator.evaluate(dst.block(0),this->time+this->time_step);
  // shift body force term to the left-hand side of the equation
  dst.block(0) *= -1.0;

  temp.equ(this->gamma0/this->time_step,src.block(0));
  temp.add(-1.0,sum_alphai_ui);
  this->mass_matrix_operator.apply_add(dst.block(0),temp);

  this->convective_operator.evaluate_add(dst.block(0),src.block(0),this->time+this->time_step);
  this->viscous_operator.evaluate_add(dst.block(0),src.block(0),this->time+this->time_step);
  this->gradient_operator.evaluate_add(dst.block(0),src.block(1),this->time+this->time_step);

  // pressure-block
  this->divergence_operator.evaluate(dst.block(1),src.block(0),this->time+this->time_step);
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

//  template<class NUMBER>
//  void output_eigenvalues(const std::vector<NUMBER> &eigenvalues,const std::string &text)
//  {
//    deallog<< text;
//    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
//      {
//        deallog<< ' ' << eigenvalues.at(j);
//      }
//    deallog << std::endl;
//  }

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                          parallel::distributed::BlockVector<value_type> const &src,
                          parallel::distributed::Vector<value_type> const      *solution_linearization)
{
  if(solution_linearization != nullptr)
  {
    velocity_linear = *solution_linearization;
  }

  ReductionControl solver_control (this->param.max_iter_linear, this->param.abs_tol_linear, this->param.rel_tol_linear);
  SolverGMRES<parallel::distributed::BlockVector<value_type> > solver (solver_control);
  NavierStokesCoupledMatrix<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> global_matrix;
  global_matrix.initialize(*this);

//    solver.connect_eigenvalues_slot(std_cxx11::bind(output_eigenvalues<std::complex<double> >,std_cxx11::_1,"Eigenvalues: "),true);

  solver.solve (global_matrix, dst, src, PreconditionIdentity());

  return solver_control.last_step();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>       &dst,
                         parallel::distributed::Vector<value_type> const      &sum_alphai_ui)
{
  evaluate_nonlinear_residual(residual,dst,sum_alphai_ui);

  value_type norm_r = residual.l2_norm();
  value_type norm_r_0 = norm_r;

  // Newton iteration
  unsigned int n_iter = 0;
  while(norm_r > this->param.abs_tol_newton && norm_r/norm_r_0 > this->param.rel_tol_newton && n_iter < this->param.max_iter_newton)
  {
    // reset increment
    increment = 0.0;

    residual *= -1.0;
    unsigned int linear_iterations = solve_linearized_problem(increment, residual, &dst.block(0));

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "  Number of linear solver iterations: " << linear_iterations << std::endl;

    // update solution
    dst.add(1.0, increment);

    evaluate_nonlinear_residual(residual,dst,sum_alphai_ui);

    norm_r = residual.l2_norm();

    ++n_iter;
  }

  if(n_iter >= this->param.max_iter_newton)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout<<"Newton solver failed to solve nonlinear problem to given tolerance. Maximum number of iterations exceeded!" << std::endl;
  }

  return n_iter;
}

#endif /* INCLUDE_DGNAVIERSTOKESCOUPLED_H_ */
