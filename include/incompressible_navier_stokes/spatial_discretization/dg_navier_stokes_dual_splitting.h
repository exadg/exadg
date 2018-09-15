/*
 * DGNavierStokesDualSplitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_

#include "../../incompressible_navier_stokes/preconditioners/multigrid_preconditioner_navier_stokes.h"
#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_projection_methods.h"
#include "../../incompressible_navier_stokes/spatial_discretization/helmholtz_operator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/pressure_neumann_bc_convective_term.h"
#include "../../incompressible_navier_stokes/spatial_discretization/pressure_neumann_bc_viscous_term.h"
#include "../../incompressible_navier_stokes/spatial_discretization/velocity_divergence_convective_term.h"
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers.h"

namespace IncNS
{
template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
class DGNavierStokesDualSplitting : public DGNavierStokesProjectionMethods<dim,
                                                                           fe_degree,
                                                                           fe_degree_p,
                                                                           fe_degree_xwall,
                                                                           xwall_quad_rule,
                                                                           Number>
{
public:
  typedef DGNavierStokesProjectionMethods<dim,
                                          fe_degree,
                                          fe_degree_p,
                                          fe_degree_xwall,
                                          xwall_quad_rule,
                                          Number>
    PROJECTION_METHODS_BASE;

  typedef typename PROJECTION_METHODS_BASE::VectorType VectorType;

  typedef DGNavierStokesDualSplitting<dim,
                                      fe_degree,
                                      fe_degree_p,
                                      fe_degree_xwall,
                                      xwall_quad_rule,
                                      Number>
    THIS;

  DGNavierStokesDualSplitting(parallel::distributed::Triangulation<dim> const & triangulation,
                              InputParameters<dim> const &                      parameter)
    : PROJECTION_METHODS_BASE(triangulation, parameter),
      fe_param(parameter),
      sum_alphai_ui(nullptr),
      evaluation_time(0.0),
      scaling_factor_time_derivative_term(1.0)
  {
  }

  virtual ~DGNavierStokesDualSplitting()
  {
  }

  void
  setup_solvers(double const & time_step_size, double const & scaling_factor_time_derivative_term);

  /*
   *  implicit solution of convective step
   */
  void
  solve_nonlinear_convective_problem(VectorType &       dst,
                                     VectorType const & sum_alphai_ui,
                                     double const &     eval_time,
                                     double const &     scaling_factor_mass_matrix_term,
                                     unsigned int &     newton_iterations,
                                     unsigned int &     linear_iterations);

  /*
   *  The implementation of the Newton solver requires that the underlying operator
   *  implements a function called "evaluate_nonlinear_residual"
   */
  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src);

  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "initialize_vector_for_newton_solver"
   */
  void
  initialize_vector_for_newton_solver(VectorType & src) const
  {
    this->initialize_vector_velocity(src);
  }

  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "set_solution_linearization"
   */
  void
  set_solution_linearization(VectorType const & solution_linearization)
  {
    velocity_linearization = solution_linearization;
  }

  /*
   *  To solve the linearized convective problem, the underlying operator
   *  has to implement a function called "vmult"
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  /*
   *  This function calculates the matrix vector product for the linearized convective problem.
   */
  void
  apply_linearized_convective_problem(VectorType & dst, VectorType const & src) const;

  // convective term
  void
  evaluate_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                         VectorType const & src,
                                                         double const       evaluation_time) const;

  // body forces
  void
  evaluate_body_force_and_apply_inverse_mass_matrix(VectorType & dst,
                                                    double const evaluation_time) const;

  // rhs pressure: velocity divergence
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  void
  rhs_velocity_divergence_term(VectorType & dst, double const & evaluation_time) const;

  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & eval_time);

  void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src);

  // rhs pressure
  void
  rhs_ppe_nbc_add(VectorType & dst, double const & evaluation_time);

  // rhs pressure: Neumann BC convective term
  void
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure: Neumann BC viscous term
  void
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const;

  // viscous step
  unsigned int
  solve_viscous(VectorType &       dst,
                VectorType const & src,
                double const &     scaling_factor_time_derivative_term);

  // apply Helmholtz operator
  void
  apply_helmholtz_operator(VectorType & dst, VectorType const & src) const;

  FEParameters<dim> const &
  get_fe_parameters() const
  {
    return this->fe_param;
  }

  // TODO
  void
  get_wall_times_projection_helmholtz_operator(double & wall_time_projection,
                                               double & wall_time_helmholtz)
  {
    if(this->param.use_divergence_penalty == true && this->param.use_continuity_penalty == true)
    {
      if(this->use_optimized_projection_operator == false)
      {
        typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim,
                                                                 fe_degree,
                                                                 fe_degree_p,
                                                                 fe_degree_xwall,
                                                                 xwall_quad_rule,
                                                                 Number>
          PROJ_OPERATOR;

        std::shared_ptr<PROJ_OPERATOR> proj_op =
          std::dynamic_pointer_cast<PROJ_OPERATOR>(this->projection_operator);
        AssertThrow(proj_op.get() != 0,
                    ExcMessage("Projection operator is not initialized correctly."));

        wall_time_projection = proj_op->get_wall_time();
      }
      // TODO
      else // use_optimized_projection_operator == true
      {
        typedef ProjectionOperatorOptimized<dim,
                                            fe_degree,
                                            fe_degree_p,
                                            fe_degree_xwall,
                                            xwall_quad_rule,
                                            Number>
          PROJ_OPERATOR;

        std::shared_ptr<PROJ_OPERATOR> proj_op =
          std::dynamic_pointer_cast<PROJ_OPERATOR>(this->projection_operator);
        AssertThrow(proj_op.get() != 0,
                    ExcMessage("Projection operator is not initialized correctly."));

        wall_time_projection = proj_op->get_wall_time();
      }
    }
    else
    {
      wall_time_projection = 0.0;
    }

    wall_time_helmholtz = this->helmholtz_operator.get_wall_time();
  }

protected:
  FEParameters<dim> fe_param;

  HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> helmholtz_operator;
  std::shared_ptr<PreconditionerBase<Number>> helmholtz_preconditioner;

private:
  // Helmholtz solver
  std::shared_ptr<IterativeSolverBase<VectorType>> helmholtz_solver;

  // Implicit solution of convective step
  VectorType         velocity_linearization;
  VectorType         temp;
  VectorType const * sum_alphai_ui;

  // implicit solution of convective step
  std::shared_ptr<InverseMassMatrixPreconditioner<dim, fe_degree, Number>>
    preconditioner_convective_problem;

  std::shared_ptr<IterativeSolverBase<VectorType>> linear_solver;
  std::shared_ptr<NewtonSolver<VectorType, THIS, THIS, IterativeSolverBase<VectorType>>>
    newton_solver;


  // rhs pressure Poisson equation: Velocity divergence term
  VelocityDivergenceConvectiveTerm<dim,
                                   fe_degree,
                                   fe_degree_p,
                                   fe_degree_xwall,
                                   xwall_quad_rule,
                                   Number>
    velocity_divergence_convective_term;

  // pressure Neumann BC
  PressureNeumannBCConvectiveTerm<dim,
                                  fe_degree,
                                  fe_degree_p,
                                  fe_degree_xwall,
                                  xwall_quad_rule,
                                  Number>
    pressure_nbc_convective_term;

  PressureNeumannBCViscousTerm<dim,
                               fe_degree,
                               fe_degree_p,
                               fe_degree_xwall,
                               xwall_quad_rule,
                               Number>
    pressure_nbc_viscous_term;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

  // setup of solvers
  void
  setup_convective_solver();

  virtual void
  setup_pressure_poisson_solver(double const time_step_size);

  void
  setup_helmholtz_solver(double const & scaling_factor_time_derivative_term);

  virtual void
  setup_helmholtz_preconditioner();

  // rhs pressure: boundary conditions for intermediate velocity u_hat
  // body force term
  void
  local_rhs_ppe_div_term_body_forces(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  local_rhs_ppe_div_term_body_forces_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  void
  local_rhs_ppe_div_term_body_forces_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // rhs pressure: NBC term
  void
  local_rhs_ppe_nbc_add(MatrixFree<dim, Number> const &               data,
                        VectorType &                                  dst,
                        VectorType const &                            src,
                        std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  local_rhs_ppe_nbc_add_face(MatrixFree<dim, Number> const &               data,
                             VectorType &                                  dst,
                             VectorType const &                            src,
                             std::pair<unsigned int, unsigned int> const & face_range) const;

  void
  local_rhs_ppe_nbc_add_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;
};

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  setup_solvers(double const & time_step_size, double const & scaling_factor_time_derivative_term)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // initialize vectors that are needed by the nonlinear solver
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    setup_convective_solver();
  }

  this->setup_pressure_poisson_solver(time_step_size);

  this->setup_projection_solver();

  setup_helmholtz_solver(scaling_factor_time_derivative_term);

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  setup_convective_solver()
{
  this->initialize_vector_velocity(temp);
  this->initialize_vector_velocity(velocity_linearization);

  // preconditioner implicit convective step
  preconditioner_convective_problem.reset(
    new InverseMassMatrixPreconditioner<dim, fe_degree, Number>(
      this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));

  // linear solver (GMRES)
  GMRESSolverData solver_data;
  solver_data.max_iter              = this->param.max_iter_linear_convective;
  solver_data.solver_tolerance_abs  = this->param.abs_tol_linear_convective;
  solver_data.solver_tolerance_rel  = this->param.rel_tol_linear_convective;
  solver_data.right_preconditioning = this->param.use_right_preconditioning_convective;
  solver_data.max_n_tmp_vectors     = this->param.max_n_tmp_vectors_convective;

  // always use inverse mass matrix preconditioner
  solver_data.use_preconditioner = true;

  // setup linear solver
  linear_solver.reset(
    new GMRESSolver<THIS, InverseMassMatrixPreconditioner<dim, fe_degree, Number>, VectorType>(
      *this, *preconditioner_convective_problem, solver_data));

  // setup Newton solver
  newton_solver.reset(new NewtonSolver<VectorType, THIS, THIS, IterativeSolverBase<VectorType>>(
    this->param.newton_solver_data_convective, *this, *this, *linear_solver));
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  setup_pressure_poisson_solver(double const time_step_size)
{
  // Call setup function of base class
  PROJECTION_METHODS_BASE::setup_pressure_poisson_solver(time_step_size);

  // RHS PPE: Velocity divergence term
  VelocityDivergenceConvectiveTermData<dim> velocity_divergence_convective_data;
  velocity_divergence_convective_data.dof_index_velocity = this->get_dof_index_velocity();
  velocity_divergence_convective_data.dof_index_pressure = this->get_dof_index_pressure();
  velocity_divergence_convective_data.bc                 = this->boundary_descriptor_velocity;

  velocity_divergence_convective_term.initialize(this->data, velocity_divergence_convective_data);

  // Pressure NBC: Convective term
  PressureNeumannBCConvectiveTermData<dim> pressure_nbc_convective_data;
  pressure_nbc_convective_data.dof_index_velocity = this->get_dof_index_velocity();
  pressure_nbc_convective_data.dof_index_pressure = this->get_dof_index_pressure();
  pressure_nbc_convective_data.bc                 = this->boundary_descriptor_pressure;

  pressure_nbc_convective_term.initialize(this->data, pressure_nbc_convective_data);

  // Pressure NBC: Viscous term
  PressureNeumannBCViscousTermData<dim> pressure_nbc_viscous_data;
  pressure_nbc_viscous_data.dof_index_velocity = this->get_dof_index_velocity();
  pressure_nbc_viscous_data.dof_index_pressure = this->get_quad_index_pressure();
  pressure_nbc_viscous_data.bc                 = this->boundary_descriptor_pressure;

  pressure_nbc_viscous_term.initialize(this->data,
                                       pressure_nbc_viscous_data,
                                       this->viscous_operator);
}


template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  setup_helmholtz_solver(double const & scaling_factor_time_derivative_term)
{
  // 1. Setup Helmholtz operator
  HelmholtzOperatorData<dim> helmholtz_operator_data;

  helmholtz_operator_data.dof_index = this->get_dof_index_velocity();

  // always unsteady problem
  helmholtz_operator_data.unsteady_problem = true;

  // set scaling factor time derivative term!
  helmholtz_operator_data.scaling_factor_time_derivative_term = scaling_factor_time_derivative_term;

  helmholtz_operator.initialize(this->data,
                                helmholtz_operator_data,
                                this->mass_matrix_operator,
                                this->viscous_operator);

  // 2. Setup Helmholtz preconditioner
  setup_helmholtz_preconditioner();

  // 3. Setup Helmholtz solver
  if(this->param.solver_viscous == SolverViscous::PCG)
  {
    // setup solver data
    CGSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
    solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }
    solver_data.update_preconditioner = this->param.update_preconditioner_viscous;

    // setup helmholtz solver
    helmholtz_solver.reset(
      new CGSolver<HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                   PreconditionerBase<Number>,
                   VectorType>(helmholtz_operator, *helmholtz_preconditioner, solver_data));
  }
  else if(this->param.solver_viscous == SolverViscous::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
    solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
    // use default value of right_preconditioning
    // use default value of max_n_tmp_vectors
    // use default value of compute_eigenvalues
    solver_data.update_preconditioner = this->param.update_preconditioner_viscous;

    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    // setup helmholtz solver
    helmholtz_solver.reset(
      new GMRESSolver<HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                      PreconditionerBase<Number>,
                      VectorType>(helmholtz_operator, *helmholtz_preconditioner, solver_data));
  }
  else if(this->param.solver_viscous == SolverViscous::FGMRES)
  {
    FGMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
    solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
    // use default value of max_n_tmp_vectors
    solver_data.update_preconditioner = this->param.update_preconditioner_viscous;

    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    helmholtz_solver.reset(
      new FGMRESSolver<HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                       PreconditionerBase<Number>,
                       VectorType>(helmholtz_operator, *helmholtz_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_viscous == SolverViscous::PCG ||
                  this->param.solver_viscous == SolverViscous::GMRES ||
                  this->param.solver_viscous == SolverViscous::FGMRES,
                ExcMessage(
                  "Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  setup_helmholtz_preconditioner()
{
  if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    helmholtz_preconditioner.reset(new InverseMassMatrixPreconditioner<dim, fe_degree, Number>(
      this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi)
  {
    helmholtz_preconditioner.reset(
      new JacobiPreconditioner<
        HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>>(
        helmholtz_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi)
  {
    helmholtz_preconditioner.reset(
      new BlockJacobiPreconditioner<
        HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>>(
        helmholtz_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_viscous;

    // use single precision for multigrid
    typedef float MultigridNumber;

    typedef MyMultigridPreconditionerVelocityDiffusion<
      dim,
      Number,
      HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, MultigridNumber>,
      HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>>
      MULTIGRID;

    helmholtz_preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(helmholtz_preconditioner);

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_u,
                                  this->mapping,
                                  /*helmholtz_operator.get_operator_data().bc->dirichlet_bc,*/
                                  (void *)&helmholtz_operator.get_operator_data());
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  solve_nonlinear_convective_problem(VectorType &       dst,
                                     VectorType const & sum_alphai_ui,
                                     double const &     eval_time,
                                     double const &     scaling_factor_mass_matrix_term,
                                     unsigned int &     newton_iterations,
                                     unsigned int &     linear_iterations)
{
  // Set sum_alphai_ui, this variable is used when evaluating the nonlinear residual
  this->sum_alphai_ui = &sum_alphai_ui;

  // set evaluation time for both the linear and the nonlinear operator
  // (=DGNavierStokesDualSplitting)
  evaluation_time = eval_time;

  // set scaling_factor_time_derivative term for both the linear and the nonlinear operator
  // (=DGNavierStokesDualSplitting)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // solve nonlinear problem
  newton_solver->solve(dst, newton_iterations, linear_iterations);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src)
{
  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst, evaluation_time);
    // shift body force term to the left-hand side of the equation
    dst *= -1.0;
  }
  else // right_hand_side == false
  {
    // set dst to zero. This is necessary since the subsequent operators
    // call functions of type ..._add
    dst = 0.0;
  }

  // temp, src, sum_alphai_ui have the same number of blocks
  temp.equ(scaling_factor_time_derivative_term, src);
  temp.add(-1.0, *sum_alphai_ui);

  this->mass_matrix_operator.apply_add(dst, temp);

  this->convective_operator.evaluate_add(dst, src, evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  vmult(VectorType & dst, VectorType const & src) const
{
  apply_linearized_convective_problem(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  apply_linearized_convective_problem(VectorType & dst, VectorType const & src) const
{
  this->mass_matrix_operator.apply(dst, src);

  dst *= scaling_factor_time_derivative_term;

  this->convective_operator.apply_linearized_add(dst,
                                                 src,
                                                 &velocity_linearization,
                                                 evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  evaluate_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                         VectorType const & src,
                                                         double const       evaluation_time) const
{
  this->convective_operator.evaluate(dst, src, evaluation_time);
  this->inverse_mass_matrix_operator->apply(dst, dst);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  evaluate_body_force_and_apply_inverse_mass_matrix(VectorType & dst,
                                                    double const evaluation_time) const
{
  this->body_force_operator.evaluate(dst, evaluation_time);

  this->inverse_mass_matrix_operator->apply(dst, dst);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_velocity_divergence_term(VectorType & dst, double const & evaluation_time) const
{
  this->divergence_operator.rhs(dst, evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & eval_time)
{
  evaluation_time = eval_time;

  VectorType src_dummy;
  this->data.loop(&THIS::local_rhs_ppe_div_term_body_forces,
                  &THIS::local_rhs_ppe_div_term_body_forces_face,
                  &THIS::local_rhs_ppe_div_term_body_forces_boundary_face,
                  this,
                  dst,
                  src_dummy);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_div_term_body_forces(MatrixFree<dim, Number> const &,
                                     VectorType &,
                                     VectorType const &,
                                     std::pair<unsigned int, unsigned int> const &) const
{
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_div_term_body_forces_face(MatrixFree<dim, Number> const &,
                                          VectorType &,
                                          VectorType const &,
                                          std::pair<unsigned int, unsigned int> const &) const
{
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_div_term_body_forces_boundary_face(
    MatrixFree<dim, Number> const & data,
    VectorType &                    dst,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FEFaceEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> fe_eval(data,
                                                                         true,
                                                                         dof_index_pressure,
                                                                         quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    fe_eval.reinit(face);

    types::boundary_id boundary_id   = data.get_boundary_id(face);
    BoundaryTypeU      boundary_type = BoundaryTypeU::Undefined;

    if(this->boundary_descriptor_velocity->dirichlet_bc.find(boundary_id) !=
       this->boundary_descriptor_velocity->dirichlet_bc.end())
    {
      boundary_type = BoundaryTypeU::Dirichlet;
    }
    else if(this->boundary_descriptor_velocity->neumann_bc.find(boundary_id) !=
            this->boundary_descriptor_velocity->neumann_bc.end())
    {
      boundary_type = BoundaryTypeU::Neumann;
    }
    else if(this->boundary_descriptor_velocity->symmetry_bc.find(boundary_id) !=
            this->boundary_descriptor_velocity->symmetry_bc.end())
    {
      boundary_type = BoundaryTypeU::Symmetry;
    }

    AssertThrow(boundary_type != BoundaryTypeU::Undefined,
                ExcMessage("Boundary type of face is invalid or not implemented."));

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

        // evaluate right-hand side
        Tensor<1, dim, VectorizedArray<Number>> rhs;
        evaluate_vectorial_function(rhs,
                                    this->field_functions->right_hand_side,
                                    q_points,
                                    evaluation_time);

        VectorizedArray<Number> flux_times_normal = rhs * fe_eval.get_normal_vector(q);
        // minus sign is introduced here which allows to call a function of type ...add()
        // and avoids a scaling of the resulting vector by the factor -1.0
        fe_eval.submit_value(-flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and Symmetry boundaries.
        // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in case
        // of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
        // prescribe a consistent boundary condition for g_{u_hat} derived from the convective step
        // of the dual splitting scheme which differs from the DBC g_u. Applying this consistent DBC
        // to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry, we obtain
        // g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous contributions for
        // g_{u_hat}*n.
        VectorizedArray<Number> zero = make_vectorized_array<Number>(0.0);
        fe_eval.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src)
{
  velocity_divergence_convective_term.calculate(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_ppe_nbc_add(VectorType & dst, double const & eval_time)
{
  evaluation_time = eval_time;

  VectorType src_dummy;
  this->data.loop(&THIS::local_rhs_ppe_nbc_add,
                  &THIS::local_rhs_ppe_nbc_add_face,
                  &THIS::local_rhs_ppe_nbc_add_boundary_face,
                  this,
                  dst,
                  src_dummy);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_nbc_add(MatrixFree<dim, Number> const &,
                        VectorType &,
                        VectorType const &,
                        std::pair<unsigned int, unsigned int> const &) const
{
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_nbc_add_face(MatrixFree<dim, Number> const &,
                             VectorType &,
                             VectorType const &,
                             std::pair<unsigned int, unsigned int> const &) const
{
}


template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  local_rhs_ppe_nbc_add_boundary_face(
    MatrixFree<dim, Number> const & data,
    VectorType &                    dst,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FEFaceEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> fe_eval(data,
                                                                         true,
                                                                         dof_index_pressure,
                                                                         quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    fe_eval.reinit(face);

    types::boundary_id boundary_id   = data.get_boundary_id(face);
    BoundaryTypeP      boundary_type = BoundaryTypeP::Undefined;

    if(this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id) !=
       this->boundary_descriptor_pressure->dirichlet_bc.end())
    {
      boundary_type = BoundaryTypeP::Dirichlet;
    }
    else if(this->boundary_descriptor_pressure->neumann_bc.find(boundary_id) !=
            this->boundary_descriptor_pressure->neumann_bc.end())
    {
      boundary_type = BoundaryTypeP::Neumann;
    }

    AssertThrow(boundary_type != BoundaryTypeP::Undefined,
                ExcMessage("Boundary type of face is invalid or not implemented."));

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

        // evaluate right-hand side
        Tensor<1, dim, VectorizedArray<Number>> rhs;
        evaluate_vectorial_function(rhs,
                                    this->field_functions->right_hand_side,
                                    q_points,
                                    evaluation_time);

        // evaluate boundary condition
        Tensor<1, dim, VectorizedArray<Number>> dudt;

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
        it = this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
        evaluate_vectorial_function(dudt, it->second, q_points, evaluation_time);

        Tensor<1, dim, VectorizedArray<Number>> normal = fe_eval.get_normal_vector(q);

        VectorizedArray<Number> h;

        h = -normal * (dudt - rhs);

        fe_eval.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        VectorizedArray<Number> zero = make_vectorized_array<Number>(0.0);
        fe_eval.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(dst);
  }
}


template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const
{
  pressure_nbc_convective_term.calculate(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const
{
  pressure_nbc_viscous_term.calculate(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
unsigned int
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  solve_viscous(VectorType & dst, VectorType const & src, double const & factor)
{
  // Update Helmholtz operator
  helmholtz_operator.set_scaling_factor_time_derivative_term(factor);

  unsigned int n_iter = helmholtz_solver->solve(dst, src);

  return n_iter;
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
  apply_helmholtz_operator(VectorType & dst, VectorType const & src) const
{
  // Update Helmholtz operator
  helmholtz_operator.vmult(dst, src);
}



} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_ \
        */
