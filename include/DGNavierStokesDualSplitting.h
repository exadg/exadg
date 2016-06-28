/*
 * DGNavierStokesDualSplitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_

#include "DGNavierStokesBase.h"

#include "HelmholtzSolver.h"
#include "ProjectionSolver.h"
#include "poisson_solver.h"
#include "CurlCompute.h"

//forward declarations
template<int dim> class RHS;
template<int dim> class NeumannBoundaryVelocity;
template<int dim> class PressureBC_dudt;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplitting : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  static const unsigned int n_actual_q_points_vel_nonlinear = (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall) ? n_q_points_1d_xwall : fe_degree+(fe_degree+2)/2;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree_p,fe_degree_xwall,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::n_actual_q_points_vel_linear,1,value_type,false> FEFaceEval_Pressure_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_nonlinear,1,value_type,false> FEFaceEval_Pressure_Velocity_nonlinear;

  DGNavierStokesDualSplitting(parallel::distributed::Triangulation<dim> const &triangulation,
                              InputParameters const                           &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter),
    projection_operator(nullptr)
  {}

  ~DGNavierStokesDualSplitting()
  {
    delete projection_operator;
    projection_operator = nullptr;
  }

  void setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs);

  // convective step
  unsigned int solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                                   parallel::distributed::Vector<value_type> const &sum_alphai_ui);

  unsigned int solve_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                                    parallel::distributed::Vector<value_type> const &src,
                                                    parallel::distributed::Vector<value_type> const *velocity_linearization);

  void evaluate_nonlinear_residual_convective_step (parallel::distributed::Vector<value_type>       &dst,
                                                    parallel::distributed::Vector<value_type> const &src,
                                                    parallel::distributed::Vector<value_type> const &sum_alphai_ui);

  void apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                            parallel::distributed::Vector<value_type> const &src) const;

  // inverse mass matrix
  void apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                                  parallel::distributed::Vector<value_type> const &src) const;

  // body forces
  void  calculate_body_force (parallel::distributed::Vector<value_type>  &dst,
                              const value_type                           evaluation_time) const;

  // rhs pressure
  void rhs_pressure_divergence_term (parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &src,
                                     const value_type                                 evaluation_time) const;

  void rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                             const parallel::distributed::Vector<value_type> &src) const;

  void rhs_pressure_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src) const;

  void rhs_pressure_viscous_term (parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src) const;

  // nullspace projection (in case of pure Dirichlet BC)
  void apply_nullspace_projection (parallel::distributed::Vector<value_type>  &dst) const;

  // solve pressure step
  unsigned int solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src) const;

  // projection step
  void rhs_projection (parallel::distributed::Vector<value_type>       &dst,
                       const parallel::distributed::Vector<value_type> &src_velocity,
                       const parallel::distributed::Vector<value_type> &src_pressure) const;

  unsigned int solve_projection (parallel::distributed::Vector<value_type>       &dst,
                                 const parallel::distributed::Vector<value_type> &src,
                                 const parallel::distributed::Vector<value_type> &velocity_n,
                                 double const                                    cfl) const;

  // viscous step
  void  rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const;

  unsigned int solve_viscous (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src);


  // initialization of vectors
  void initialize_vector_pressure(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));
  }

private:
  PoissonSolver<dim> pressure_poisson_solver;
  LaplaceOperator<dim,value_type> laplace_operator;

  std_cxx11::shared_ptr<ProjectionSolverBase<value_type> > projection_solver;
  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> * projection_operator;

  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> helmholtz_operator;
  HelmholtzSolver<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> helmholtz_solver;

  parallel::distributed::Vector<value_type> velocity_linear;
  parallel::distributed::Vector<value_type> residual_convective_step;
  parallel::distributed::Vector<value_type> increment_convective_step;
  parallel::distributed::Vector<value_type> temp;

  // rhs pressure: BC term
  void local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                &data,
                                   parallel::distributed::Vector<value_type>       &dst,
                                   const parallel::distributed::Vector<value_type> &src,
                                   const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                 &data,
                                        parallel::distributed::Vector<value_type>        &dst,
                                        const parallel::distributed::Vector<value_type>  &src,
                                        const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_BC_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                parallel::distributed::Vector<value_type>        &dst,
                                                const parallel::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: convective term
  void local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                &data,
                                           parallel::distributed::Vector<value_type>       &dst,
                                           const parallel::distributed::Vector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                 &data,
                                                parallel::distributed::Vector<value_type>        &dst,
                                                const parallel::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_convective_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                        parallel::distributed::Vector<value_type>        &dst,
                                                        const parallel::distributed::Vector<value_type>  &src,
                                                        const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: viscous term
  void local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                 &data,
                                             parallel::distributed::Vector<value_type>        &dst,
                                             const parallel::distributed::Vector<value_type>  &src,
                                             const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_viscous_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                     parallel::distributed::Vector<value_type>        &dst,
                                                     const parallel::distributed::Vector<value_type>  &src,
                                                     const std::pair<unsigned int,unsigned int>       &face_range) const;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs)
{
  // initialize vectors that are needed by the nonlinear solver
  if(this->param.solve_stokes_equations == false && this->param.convective_step_implicit == true)
  {
    this->initialize_vector_velocity(velocity_linear);
    this->initialize_vector_velocity(residual_convective_step);
    this->initialize_vector_velocity(increment_convective_step);
    this->initialize_vector_velocity(temp);
  }

  // Laplace Operator
  LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.laplace_dof_index = static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure);
  laplace_operator_data.laplace_quad_index = static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::pressure);
  laplace_operator_data.penalty_factor = this->param.IP_factor_pressure;

  // TODO
  /*
   * approach of Ferrer et al.: increase penalty parameter when reducing the time step
   * in order to improve stability in the limit of small time steps
   */

  /*
  double dt_ref = 0.1;
  laplace_operator_data.penalty_factor = this->param.IP_factor_pressure/time_step*dt_ref;
  */
  laplace_operator_data.dirichlet_boundaries = this->neumann_boundary;
  laplace_operator_data.neumann_boundaries = this->dirichlet_boundary;
  laplace_operator_data.periodic_face_pairs_level0 = periodic_face_pairs;
  laplace_operator.reinit(this->data,this->mapping,laplace_operator_data);

  // Pressure Poisson solver
  PoissonSolverData poisson_solver_data;
  poisson_solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
  poisson_solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
  poisson_solver_data.solver_poisson = this->param.solver_poisson;
  poisson_solver_data.preconditioner_poisson = this->param.preconditioner_poisson;
  poisson_solver_data.multigrid_smoother = this->param.multigrid_smoother;
  poisson_solver_data.coarse_solver = this->param.multigrid_coarse_grid_solver;
  pressure_poisson_solver.initialize(laplace_operator,this->mapping,this->data,poisson_solver_data);

  // initialize projection solver
  ProjectionOperatorData projection_operator_data;
  projection_operator_data.penalty_parameter_divergence = this->param.penalty_factor_divergence;
  projection_operator_data.penalty_parameter_continuity = this->param.penalty_factor_continuity;
  projection_operator_data.solve_stokes_equations = this->param.solve_stokes_equations;

  if(this->param.projection_type == ProjectionType::NoPenalty)
  {
    projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, value_type>(this->data,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity)));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::LU)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
        projection_operator_data);

    projection_solver.reset(new DirectProjectionSolverDivergencePenalty
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(projection_operator));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::PCG)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergencePenalty
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                              projection_operator,
                              projection_solver_data));
  }
  else if(this->param.projection_type == ProjectionType::DivergenceAndContinuityPenalty)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                              projection_operator,
                              projection_solver_data));
  }

  // helmholtz operator
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity);
  helmholtz_operator_data.formulation_viscous_term = this->param.formulation_viscous_term;
  helmholtz_operator_data.IP_formulation_viscous = this->param.IP_formulation_viscous;
  helmholtz_operator_data.IP_factor_viscous = this->param.IP_factor_viscous;
  helmholtz_operator_data.dirichlet_boundaries = this->dirichlet_boundary;
  helmholtz_operator_data.neumann_boundaries = this->neumann_boundary;
  helmholtz_operator_data.viscosity = this->viscosity;
  helmholtz_operator_data.mass_matrix_coefficient = this->gamma0/this->time_step;
  helmholtz_operator.reinit(this->data, this->mapping, helmholtz_operator_data,this->fe_param);

  HelmholtzSolverData helmholtz_solver_data;
  helmholtz_solver_data.solver_viscous = this->param.solver_viscous;
  helmholtz_solver_data.preconditioner_viscous = this->param.preconditioner_viscous;
  helmholtz_solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
  helmholtz_solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
  helmholtz_solver_data.multigrid_smoother = this->param.multigrid_smoother_viscous;
  helmholtz_solver_data.coarse_solver = this->param.multigrid_coarse_grid_solver_viscous;

  helmholtz_solver.initialize(helmholtz_operator, this->mapping, this->data, helmholtz_solver_data,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
      this->fe_param);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
struct LinearizedConvectionMatrix : public Subscriptor
{
  void initialize(DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
  {
    ns_operation = &ns_op;
  }
  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    ns_operation->apply_linearized_convective_problem(dst,src);
  }
  DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                     parallel::distributed::Vector<value_type> const &src) const
{
  this->mass_matrix_operator.apply(dst,src);
  // dst-vector only contains velocity (and not the pressure)
  dst *= this->gamma0/this->time_step;

  this->convective_operator.apply_linearized_add(dst,src,&velocity_linear,this->time+this->time_step);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>        &dst,
                                    parallel::distributed::Vector<value_type> const  &sum_alphai_ui)
{
  evaluate_nonlinear_residual_convective_step(residual_convective_step,dst,sum_alphai_ui);

  value_type norm_r = residual_convective_step.l2_norm();
  value_type norm_r_0 = norm_r;

  // Newton iteration
  unsigned int n_iter = 0;
  while(norm_r > this->param.abs_tol_newton && norm_r/norm_r_0 > this->param.rel_tol_newton && n_iter < this->param.max_iter_newton)
  {
    // reset increment
    increment_convective_step = 0.0;

    residual_convective_step *= -1.0;

    // solve linearized problem
    solve_linearized_convective_problem(increment_convective_step,residual_convective_step,&dst);

    // update solution
    dst.add(1.0, increment_convective_step);

    // calculate residual of nonlinear equation
    evaluate_nonlinear_residual_convective_step(residual_convective_step,dst,sum_alphai_ui);

    norm_r = residual_convective_step.l2_norm();
    ++n_iter;
  }

  if(n_iter >= this->param.max_iter_newton)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout<<"Newton solver failed to solve nonlinear convective problem to given tolerance. Maximum number of iterations exceeded!" << std::endl;
  }

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual_convective_step (parallel::distributed::Vector<value_type>             &dst,
                                             const parallel::distributed::Vector<value_type>       &src,
                                             const parallel::distributed::Vector<value_type>       &sum_alphai_ui)
{
  this->body_force_operator.evaluate(dst,this->time+this->time_step);
  // shift body force term to the left-hand side of the equation
  dst *= -1.0;

  // temp, src, sum_alphai_ui have the same number of blocks
  temp.equ(this->gamma0/this->time_step,src);
  temp.add(-1.0,sum_alphai_ui);

  this->mass_matrix_operator.apply_add(dst,temp);

  this->convective_operator.evaluate_add(dst,src,this->time+this->time_step);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                     parallel::distributed::Vector<value_type> const &src,
                                     parallel::distributed::Vector<value_type> const *velocity_linearization)
{
  velocity_linear = *velocity_linearization;

  ReductionControl solver_control_conv (this->param.max_iter_linear, this->param.abs_tol_linear, this->param.rel_tol_linear);
  SolverGMRES<parallel::distributed::Vector<value_type> > linear_solver_conv (solver_control_conv);
  LinearizedConvectionMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
    linearized_convection_matrix;
  linearized_convection_matrix.initialize(*this);
  InverseMassMatrixPreconditioner<dim,fe_degree,value_type> preconditioner_conv(this->data,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));
  try
  {
    linear_solver_conv.solve (linearized_convection_matrix, dst, src, preconditioner_conv); //PreconditionIdentity());
    /*
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Linear solver:" << std::endl;
      std::cout << "  Number of iterations: " << solver_control_conv.last_step() << std::endl;
      std::cout << "  Initial value: " << solver_control_conv.initial_value() << std::endl;
      std::cout << "  Last value: " << solver_control_conv.last_value() << std::endl << std::endl;
    }
    */
  }
  catch (SolverControl::NoConvergence &)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout << "Linear solver of convective step failed to solve to given tolerance." << std::endl;
  }
  return solver_control_conv.last_step();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &src) const
{
  this->inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
calculate_body_force (parallel::distributed::Vector<value_type>  &dst,
                      const value_type                           evaluation_time) const
{
  this->body_force_operator.evaluate(dst,evaluation_time);

  this->inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                const parallel::distributed::Vector<value_type>  &src) const
{
  unsigned int n_iter = pressure_poisson_solver.solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_nullspace_projection (parallel::distributed::Vector<value_type>  &dst) const
{
  pressure_poisson_solver.get_matrix().apply_nullspace_projection(dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_divergence_term (parallel::distributed::Vector<value_type>        &dst,
                              const parallel::distributed::Vector<value_type>  &src,
                              const value_type                                 evaluation_time) const
{
  this->divergence_operator.evaluate(dst,src,evaluation_time);

  if(this->param.small_time_steps_stability == true)
    dst *= -1.0;
  else
    dst *= -this->gamma0/this->time_step;

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                      const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                 &,
                            parallel::distributed::Vector<value_type>        &,
                            const parallel::distributed::Vector<value_type>  &,
                            const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::Vector<value_type>        &dst,
                                          const parallel::distributed::Vector<value_type>  &,
                                          const std::pair<unsigned int,unsigned int>       &face_range) const
{
  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

  //TODO: quadrature formula
//    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,
//        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_pressure.reinit (face);

    double factor = pressure_poisson_solver.get_matrix().get_penalty_factor();
    VectorizedArray<value_type> tau_IP = fe_eval_pressure.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      if (this->dirichlet_boundary.find(data.get_boundary_indicator(face)) != this->dirichlet_boundary.end())
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);

        Tensor<1,dim,VectorizedArray<value_type> > dudt_np, rhs_np;
        PressureBC_dudt<dim> neumann_boundary_pressure(this->time+this->time_step);
        RHS<dim> f(this->time+this->time_step);
        for(unsigned int d=0;d<dim;++d)
        {
          value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
          value_type array_f [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array_dudt[n] = neumann_boundary_pressure.value(q_point,d);
            array_f[n] = f.value(q_point,d);
          }
          dudt_np[d].load(&array_dudt[0]);
          rhs_np[d].load(&array_f[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
        VectorizedArray<value_type> h;

        h = - normal * (dudt_np - rhs_np);

        fe_eval_pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        fe_eval_pressure.submit_value(h,q);
      }
      else if (this->neumann_boundary.find(data.get_boundary_indicator(face)) != this->neumann_boundary.end())
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
        VectorizedArray<value_type> g;

        AnalyticalSolution<dim> dirichlet_boundary(false,this->time+this->time_step);
        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = dirichlet_boundary.value(q_point);
        }
        g.load(&array[0]);

        fe_eval_pressure.submit_normal_gradient(-g,q);
        fe_eval_pressure.submit_value(2.0 * tau_IP * g,q);
      }
    }
    fe_eval_pressure.integrate(true,true);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_convective_term (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type>  &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                &,
                                    parallel::distributed::Vector<value_type>       &,
                                    const parallel::distributed::Vector<value_type> &,
                                    const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                &,
                                         parallel::distributed::Vector<value_type>       &,
                                         const parallel::distributed::Vector<value_type> &,
                                         const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                                  parallel::distributed::Vector<value_type>        &dst,
                                                  const parallel::distributed::Vector<value_type>  &src,
                                                  const std::pair<unsigned int,unsigned int>       &face_range) const
{

  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,this->fe_param,true,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_velocity.reinit(face);
    fe_eval_velocity.read_dof_values(src);
    fe_eval_velocity.evaluate (true,true);

    fe_eval_pressure.reinit (face);

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      if (this->dirichlet_boundary.find(data.get_boundary_indicator(face)) != this->dirichlet_boundary.end())
      {
        VectorizedArray<value_type> h;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > grad_u = fe_eval_velocity.get_gradient(q);
        Tensor<1,dim,VectorizedArray<value_type> > convective_term = grad_u * u + fe_eval_velocity.get_divergence(q) * u;

        h = - normal * convective_term;

        fe_eval_pressure.submit_value(h,q);
      }
      else if (this->neumann_boundary.find(data.get_boundary_indicator(face)) != this->neumann_boundary.end())
      {
        fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
      }
    }
    fe_eval_pressure.integrate(true,false);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_viscous_term (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                 &,
                                      parallel::distributed::Vector<value_type>        &,
                                      const parallel::distributed::Vector<value_type>  &,
                                      const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term_boundary_face (const MatrixFree<dim,value_type>                &data,
                                               parallel::distributed::Vector<value_type>       &dst,
                                               const parallel::distributed::Vector<value_type> &src,
                                               const std::pair<unsigned int,unsigned int>      &face_range) const
{
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_omega(data,this->fe_param,true,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_pressure.reinit (face);

    fe_eval_omega.reinit (face);
    fe_eval_omega.read_dof_values(src);
    fe_eval_omega.evaluate (false,true);

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      VectorizedArray<value_type> viscosity;
      if(this->viscous_operator.viscosity_is_variable())
        viscosity = this->viscous_operator.get_viscous_coefficient_face()[face][q];
      else
        viscosity = make_vectorized_array<value_type>(this->viscous_operator.get_const_viscosity());

      if (this->dirichlet_boundary.find(data.get_boundary_indicator(face)) != this->dirichlet_boundary.end())
      {
        VectorizedArray<value_type> h;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > curl_omega = CurlCompute<dim,FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> >::compute(fe_eval_omega,q);
        h = - normal * (viscosity*curl_omega);

        fe_eval_pressure.submit_value(h,q);
      }
      else if (this->neumann_boundary.find(data.get_boundary_indicator(face)) != this->neumann_boundary.end())
      {
        fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
      }
    }
    fe_eval_pressure.integrate(true,false);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_projection (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const parallel::distributed::Vector<value_type> &velocity_n,
                  double const                                    cfl) const
{
  if(this->param.projection_type != ProjectionType::NoPenalty)
    projection_operator->calculate_array_penalty_parameter(velocity_n,cfl,this->time_step);

  unsigned int n_iter = projection_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_projection (parallel::distributed::Vector<value_type>        &dst,
                const parallel::distributed::Vector<value_type>  &src_velocity,
                const parallel::distributed::Vector<value_type>  &src_pressure) const
{
  this->gradient_operator.evaluate(dst,src_pressure,this->time+this->time_step);

  dst *= -this->time_step/this->gamma0;

  this->mass_matrix_operator.apply_add(dst,src_velocity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_viscous (parallel::distributed::Vector<value_type>       &dst,
               const parallel::distributed::Vector<value_type> &src)
{
  helmholtz_operator.set_mass_matrix_coefficient(this->gamma0/this->time_step);
  // viscous_operator.set_constant_viscosity(viscosity);
  // viscous_operator.set_variable_viscosity(viscosity);
  unsigned int n_iter = helmholtz_solver.solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
             const parallel::distributed::Vector<value_type> &src) const
{
  this->mass_matrix_operator.apply(dst,src);
  dst *= this->gamma0/this->time_step;

  this->viscous_operator.rhs_add(dst,this->time+this->time_step);
}


#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_ */
