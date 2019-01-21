/*
 * dg_navier_stokes_pressure_correction.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_pressure_correction.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::DGNavierStokesPressureCorrection(
  parallel::distributed::Triangulation<dim> const & triangulation,
  InputParameters<dim> const &                      parameters_in,
  std::shared_ptr<Postprocessor>                    postprocessor_in)
  : PROJECTION_METHODS_BASE(triangulation, parameters_in, postprocessor_in),
    rhs_vector(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0)
{
}

template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  ~DGNavierStokesPressureCorrection()
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::setup_solvers(
  double const & scaling_factor_time_derivative_term)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  setup_momentum_solver(scaling_factor_time_derivative_term);

  this->setup_pressure_poisson_solver();

  this->setup_projection_solver();

  setup_inverse_mass_matrix_operator_pressure();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::setup_momentum_solver(
  double const & scaling_factor_time_derivative_term)
{
  initialize_momentum_operator(scaling_factor_time_derivative_term);

  initialize_momentum_preconditioner();

  initialize_momentum_solver();
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::initialize_momentum_operator(
  double const & scaling_factor_time_derivative_term)
{
  MomentumOperatorData<dim> momentum_operator_data;

  // unsteady problem
  momentum_operator_data.unsteady_problem = true;

  momentum_operator_data.scaling_factor_time_derivative_term = scaling_factor_time_derivative_term;

  // convective problem
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    momentum_operator_data.convective_problem = true;
  }
  else
  {
    momentum_operator_data.convective_problem = false;
  }

  momentum_operator_data.dof_index       = this->get_dof_index_velocity();
  momentum_operator_data.quad_index_std  = this->get_quad_index_velocity_linear();
  momentum_operator_data.quad_index_over = this->get_quad_index_velocity_nonlinear();

  momentum_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  momentum_operator_data.viscous_operator_data     = this->viscous_operator_data;
  momentum_operator_data.convective_operator_data  = this->convective_operator_data;

  momentum_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  momentum_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;
  momentum_operator_data.mg_operator_type = this->param.multigrid_operator_type_momentum;

  momentum_operator.reinit(this->get_data(),
                           momentum_operator_data,
                           this->mass_matrix_operator,
                           this->viscous_operator,
                           this->convective_operator);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  initialize_momentum_preconditioner()
{
  if(this->param.preconditioner_momentum == MomentumPreconditioner::InverseMassMatrix)
  {
    momentum_preconditioner.reset(new InverseMassMatrixPreconditioner<dim, degree_u, Number, dim>(
      this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::PointJacobi)
  {
    momentum_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperator<dim, degree_u, Number>>(momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::BlockJacobi)
  {
    momentum_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, degree_u, Number>>(momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::Multigrid)
  {
    typedef float MultigridNumber;

    typedef MultigridPreconditioner<dim, degree_u, Number, MultigridNumber> MULTIGRID;

    momentum_preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(momentum_preconditioner);


    auto &                               dof_handler = this->get_dof_handler_u();
    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  momentum_operator.get_operator_data());
  }
  else
  {
    AssertThrow(this->param.preconditioner_momentum == MomentumPreconditioner::None,
                ExcNotImplemented());
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::initialize_momentum_solver()
{
  if(this->param.solver_momentum == SolverMomentum::CG)
  {
    // setup solver data
    CGSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver.reset(
      new CGSolver<MomentumOperator<dim, degree_u, Number>, PreconditionerBase<Number>, VectorType>(
        momentum_operator, *momentum_preconditioner, solver_data));
  }
  else if(this->param.solver_momentum == SolverMomentum::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    solver_data.compute_eigenvalues  = false;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver.reset(
      new GMRESSolver<MomentumOperator<dim, degree_u, Number>,
                      PreconditionerBase<Number>,
                      VectorType>(momentum_operator, *momentum_preconditioner, solver_data));
  }
  else if(this->param.solver_momentum == SolverMomentum::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    momentum_linear_solver.reset(
      new FGMRESSolver<MomentumOperator<dim, degree_u, Number>,
                       PreconditionerBase<Number>,
                       VectorType>(momentum_operator, *momentum_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for momentum equation is not implemented."));
  }


  // Navier-Stokes equations with an implicit treatment of the convective term
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    // initialize temp vector
    this->initialize_vector_velocity(temp_vector);

    // setup Newton solver
    momentum_newton_solver.reset(new NewtonSolver<VectorType,
                                                  THIS,
                                                  MomentumOperator<dim, degree_u, Number>,
                                                  IterativeSolverBase<VectorType>>(
      this->param.newton_solver_data_momentum, *this, momentum_operator, *momentum_linear_solver));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  setup_inverse_mass_matrix_operator_pressure()
{
  // inverse mass matrix operator pressure (needed for pressure update in case of rotational
  // formulation)
  inverse_mass_matrix_operator_pressure.initialize(this->data,
                                                   this->get_dof_index_pressure(),
                                                   this->get_quad_index_pressure());
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::solve_linear_momentum_equation(
  VectorType &       solution,
  VectorType const & rhs,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass_matrix_term,
  unsigned int &     linear_iterations)
{
  // Set scaling_factor_time_derivative_term for linear operator (=momentum_operator).
  momentum_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the momentum_operator
  // in this because because this function is only called if the convective term is not considered
  // in the momentum_operator (Stokes eq. or explicit treatment of convective term).

  linear_iterations = momentum_linear_solver->solve(solution, rhs, update_preconditioner);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::evaluate_add_body_force_term(
  VectorType & dst,
  double const evaluation_time) const
{
  this->body_force_operator.evaluate_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::rhs_add_viscous_term(
  VectorType & dst,
  double const evaluation_time) const
{
  PROJECTION_METHODS_BASE::do_rhs_add_viscous_term(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  initialize_vector_for_newton_solver(VectorType & src) const
{
  this->initialize_vector_velocity(src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     eval_time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass_matrix_term,
                                    unsigned int &     newton_iterations,
                                    unsigned int &     linear_iterations)
{
  // Set rhs_vector, this variable is used when evaluating the nonlinear residual
  this->rhs_vector = &rhs_vector;

  // Set evaluation_time for nonlinear operator (=DGNavierStokesPressureCorrection)
  evaluation_time = eval_time;
  // Set scaling_time_derivative_term for nonlinear operator (=DGNavierStokesPressureCorrection)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // Set correct evaluation time for linear operator (=momentum_operator).
  momentum_operator.set_evaluation_time(eval_time);
  // Set scaling_factor_time_derivative_term for linear operator (=momentum_operator).
  momentum_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  momentum_newton_solver->solve(dst,
                                newton_iterations,
                                linear_iterations,
                                update_preconditioner,
                                this->param.update_preconditioner_momentum_every_newton_iter);

  // Reset rhs_vector
  this->rhs_vector = nullptr;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::evaluate_nonlinear_residual(
  VectorType &       dst,
  VectorType const & src)
{
  // set dst to zero
  dst = 0.0;

  // mass matrix term
  temp_vector.equ(scaling_factor_time_derivative_term, src);
  this->mass_matrix_operator.apply_add(dst, temp_vector);

  // always evaluate convective term since this function is only called
  // if a nonlinear problem has to be solved, i.e., if the convective operator
  // has to be considered
  this->convective_operator.evaluate_add(dst, src, evaluation_time);

  // viscous term
  this->viscous_operator.evaluate_add(dst, src, evaluation_time);

  // rhs vector
  dst.add(-1.0, *rhs_vector);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     evaluation_time) const
{
  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst_u, evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.equation_type == EquationType::NavierStokes)
    this->convective_operator.evaluate_add(dst_u, src_u, evaluation_time);

  this->viscous_operator.evaluate_add(dst_u, src_u, evaluation_time);

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, evaluation_time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::apply_momentum_operator(
  VectorType &       dst,
  VectorType const & src,
  VectorType const & solution_linearization)
{
  momentum_operator.set_solution_linearization(solution_linearization);
  momentum_operator.vmult(dst, src);
}


template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::rhs_pressure_gradient_term(
  VectorType & dst,
  double const evaluation_time) const
{
  this->gradient_operator.rhs(dst, evaluation_time);
}


template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::
  apply_inverse_pressure_mass_matrix(VectorType & dst, VectorType const & src) const
{
  inverse_mass_matrix_operator_pressure.apply(dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
unsigned int
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::solve_pressure(
  VectorType &       dst,
  VectorType const & src) const
{
  return PROJECTION_METHODS_BASE::do_solve_pressure(dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::rhs_ppe_laplace_add(
  VectorType &   dst,
  double const & evaluation_time) const
{
  PROJECTION_METHODS_BASE::do_rhs_ppe_laplace_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::do_postprocessing(
  VectorType const & velocity,
  VectorType const & pressure,
  double const       time,
  unsigned int const time_step_number) const
{
  bool const standard = true;
  if(standard)
  {
    this->postprocessor->do_postprocessing(velocity,
                                           velocity, // intermediate_velocity
                                           pressure,
                                           time,
                                           time_step_number);
  }
  else // consider velocity and pressure errors instead
  {
    VectorType velocity_error;
    this->initialize_vector_velocity(velocity_error);

    VectorType pressure_error;
    this->initialize_vector_pressure(pressure_error);

    this->prescribe_initial_conditions(velocity_error, pressure_error, time);

    velocity_error.add(-1.0, velocity);
    pressure_error.add(-1.0, pressure);

    this->postprocessor->do_postprocessing(velocity_error, // error!
                                           velocity,       // intermediate_velocity
                                           pressure_error, // error!
                                           time,
                                           time_step_number);
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>::do_postprocessing_steady_problem(
  VectorType const & velocity,
  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity,
                                         velocity, // intermediate_velocity
                                         pressure);
}

} // namespace IncNS

#include "dg_navier_stokes_pressure_correction.hpp"
