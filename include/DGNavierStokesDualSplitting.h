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

#include "NewtonSolver.h"

//forward declarations
template<int dim> class RHS;
template<int dim> class NeumannBoundaryVelocity;
template<int dim> class PressureBC_dudt;

struct LinearizedConvectiveSolverData
{
  double abs_tol;
  double rel_tol;
  unsigned int max_iter;
};

template<int dim, int fe_degree, typename value_type, typename Operator>
class SolverLinearizedConvectiveProblem
{
public:
  void initialize(LinearizedConvectiveSolverData solver_data_in,
                  Operator                       *underlying_operator_in)
  {
    solver_data = solver_data_in;
    underlying_operator = underlying_operator_in;
  }

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     parallel::distributed::Vector<value_type> const &src,
                     parallel::distributed::Vector<value_type> const *solution_linearization = nullptr)
  {
    if(solution_linearization != nullptr)
      underlying_operator->set_solution_linearization(solution_linearization);

    ReductionControl solver_control (solver_data.max_iter, solver_data.abs_tol, solver_data.rel_tol);
    SolverGMRES<parallel::distributed::Vector<value_type> > solver (solver_control);
    InverseMassMatrixPreconditioner<dim,fe_degree,value_type> preconditioner(underlying_operator->get_data(),
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity));

    try
    {
      solver.solve (*underlying_operator, dst, src, preconditioner); //PreconditionIdentity());
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
    return solver_control.last_step();
  }

private:
  LinearizedConvectiveSolverData solver_data;
  Operator *underlying_operator;
};

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

  void setup_solvers ();

  // convective step
  unsigned int solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                                   parallel::distributed::Vector<value_type> const &sum_alphai_ui);

  unsigned int solve_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                                    parallel::distributed::Vector<value_type> const &src,
                                                    parallel::distributed::Vector<value_type> const *velocity_linearization);

  void evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &src);

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const;

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
  void initialize_vector_for_newton_solver(parallel::distributed::Vector<value_type> &src) const
  {
    this->initialize_vector_velocity(src);
  }

  void set_solution_linearization(parallel::distributed::Vector<value_type> const *solution_linearization)
  {
    velocity_linear = *solution_linearization;
  }

private:
  PoissonSolver<dim> pressure_poisson_solver;
  LaplaceOperator<dim,value_type> laplace_operator;

  std_cxx11::shared_ptr<ProjectionSolverBase<value_type> > projection_solver;
  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> * projection_operator;

  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> helmholtz_operator;
  HelmholtzSolver<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> helmholtz_solver;

  parallel::distributed::Vector<value_type> velocity_linear;
  parallel::distributed::Vector<value_type> temp;
  parallel::distributed::Vector<value_type> sum_alphai_ui;

  SolverLinearizedConvectiveProblem<dim,fe_degree, value_type,DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > linear_solver;
  NewtonSolver<parallel::distributed::Vector<value_type>, DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>, SolverLinearizedConvectiveProblem<dim,fe_degree,value_type,DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > > newton_solver;

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
setup_solvers ()
{
  // initialize vectors that are needed by the nonlinear solver
  if(this->param.equation_type == EquationType::NavierStokes
      && this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    this->initialize_vector_velocity(velocity_linear);
    this->initialize_vector_velocity(temp);
    this->initialize_vector_velocity(sum_alphai_ui);

    // linear solver that is used to solve the linear Stokes problem and the linearized Navier-Stokes problem
    LinearizedConvectiveSolverData linear_solver_data;
    linear_solver_data.abs_tol = this->param.abs_tol_linear;
    linear_solver_data.rel_tol = this->param.rel_tol_linear;
    linear_solver_data.max_iter = this->param.max_iter_linear;

    linear_solver.initialize(linear_solver_data,this);

    // Newton solver
    NewtonSolverData newton_solver_data;
    newton_solver_data.abs_tol = this->param.abs_tol_newton;
    newton_solver_data.rel_tol = this->param.rel_tol_newton;
    newton_solver_data.max_iter = this->param.max_iter_newton;

    newton_solver.initialize(newton_solver_data,this,&linear_solver);
  }


  // Laplace Operator
  LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.laplace_dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  laplace_operator_data.laplace_quad_index = static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure);
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
  laplace_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
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
  projection_operator_data.solve_stokes_equations = (this->param.equation_type == EquationType::Stokes);

  if(this->param.projection_type == ProjectionType::NoPenalty)
  {
    projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, value_type>(this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity)));
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
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
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
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
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
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
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

  helmholtz_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  helmholtz_operator_data.viscous_operator_data = this->viscous_operator_data;

  helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  helmholtz_operator_data.mass_matrix_coefficient = this->scaling_factor_time_derivative_term;
  helmholtz_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;

  helmholtz_operator.initialize(this->data,helmholtz_operator_data,this->mass_matrix_operator,this->viscous_operator);

  HelmholtzSolverData helmholtz_solver_data;
  helmholtz_solver_data.solver_viscous = this->param.solver_viscous;
  helmholtz_solver_data.preconditioner_viscous = this->param.preconditioner_viscous;
  helmholtz_solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
  helmholtz_solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
  helmholtz_solver_data.multigrid_smoother = this->param.multigrid_smoother_viscous;
  helmholtz_solver_data.coarse_solver = this->param.multigrid_coarse_grid_solver_viscous;

  helmholtz_solver.initialize(helmholtz_operator, this->mapping, this->data, helmholtz_solver_data,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
      this->fe_param);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
vmult (parallel::distributed::Vector<value_type>       &dst,
       parallel::distributed::Vector<value_type> const &src) const
{
  apply_linearized_convective_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                     parallel::distributed::Vector<value_type> const &src) const
{
  this->mass_matrix_operator.apply(dst,src);
  // dst-vector only contains velocity (and not the pressure)
  dst *= this->scaling_factor_time_derivative_term;

  this->convective_operator.apply_linearized_add(dst,src,&velocity_linear,this->time+this->time_step);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>        &dst,
                                    parallel::distributed::Vector<value_type> const  &sum_alphai_ui)
{
  this->sum_alphai_ui = sum_alphai_ui;
  return newton_solver.solve(dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>             &dst,
                             const parallel::distributed::Vector<value_type>       &src)
{
  this->body_force_operator.evaluate(dst,this->time+this->time_step);
  // shift body force term to the left-hand side of the equation
  dst *= -1.0;

  // temp, src, sum_alphai_ui have the same number of blocks
  temp.equ(this->scaling_factor_time_derivative_term,src);
  temp.add(-1.0,sum_alphai_ui);

  this->mass_matrix_operator.apply_add(dst,temp);

  this->convective_operator.evaluate_add(dst,src,this->time+this->time_step);
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
    dst *= -this->scaling_factor_time_derivative_term;

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
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

  //TODO: quadrature formula
//    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,
//        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

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
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

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
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

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

  dst *= - 1.0/this->scaling_factor_time_derivative_term;

  this->mass_matrix_operator.apply_add(dst,src_velocity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_viscous (parallel::distributed::Vector<value_type>       &dst,
               const parallel::distributed::Vector<value_type> &src)
{
  helmholtz_operator.set_mass_matrix_coefficient(this->scaling_factor_time_derivative_term);
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
  dst *= this->scaling_factor_time_derivative_term;

  this->viscous_operator.rhs_add(dst,this->time+this->time_step);
}


#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_ */
