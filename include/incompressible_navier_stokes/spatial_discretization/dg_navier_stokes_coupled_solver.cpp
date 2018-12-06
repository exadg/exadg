/*
 * dg_navier_stokes_coupled_solver.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_coupled_solver.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::DGNavierStokesCoupled(
  parallel::distributed::Triangulation<dim> const & triangulation,
  InputParameters<dim> const &                      parameters_in,
  std::shared_ptr<Postprocessor>                    postprocessor_in)
  : BASE(triangulation, parameters_in, postprocessor_in),
    sum_alphai_ui(nullptr),
    vector_linearization(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0),
    scaling_factor_continuity(1.0)
{
}

template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::~DGNavierStokesCoupled()
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::setup(
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in,
  std::shared_ptr<AnalyticalSolution<dim>> const  analytical_solution_in)
{
  BASE::setup(periodic_face_pairs,
              boundary_descriptor_velocity_in,
              boundary_descriptor_pressure_in,
              field_functions_in,
              analytical_solution_in);

  this->initialize_vector_velocity(temp_vector);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::setup_solvers(
  double const & scaling_factor_time_derivative_term)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  /*
   * Setup velocity convection-diffusion operator in function setup_solvers() since velocity
   * convection-diffusion operator data needs scaling_factor_time_derivative_term as input
   * parameter. Note also that the velocity_conv_diff_operator has to be initialized before calling
   * the setup of the BlockPreconditioner!
   */
  initialize_momentum_operator(scaling_factor_time_derivative_term);

  initialize_preconditioner_coupled();

  initialize_solver_coupled();

  this->setup_projection_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::initialize_momentum_operator(
  double const & scaling_factor_time_derivative_term)
{
  MomentumOperatorData<dim> momentum_operator_data;

  momentum_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  momentum_operator_data.viscous_operator_data     = this->viscous_operator_data;
  momentum_operator_data.convective_operator_data  = this->convective_operator_data;

  // unsteady problem
  if(unsteady_problem_has_to_be_solved())
    momentum_operator_data.unsteady_problem = true;
  else
    momentum_operator_data.unsteady_problem = false;

  momentum_operator_data.scaling_factor_time_derivative_term = scaling_factor_time_derivative_term;

  // convective problem
  if(nonlinear_problem_has_to_be_solved())
    momentum_operator_data.convective_problem = true;
  else
    momentum_operator_data.convective_problem = false;

  momentum_operator_data.dof_index      = this->get_dof_index_velocity();
  momentum_operator_data.quad_index_std = this->get_quad_index_velocity_linear();

  momentum_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  momentum_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;
  momentum_operator_data.mg_operator_type = this->param.momentum_multigrid_operator_type;

  momentum_operator.reinit(this->get_data(),
                           momentum_operator_data,
                           this->mass_matrix_operator,
                           this->viscous_operator,
                           this->convective_operator);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::initialize_preconditioner_coupled()
{
  // setup preconditioner
  if(this->param.preconditioner_linearized_navier_stokes ==
       PreconditionerLinearizedNavierStokes::BlockDiagonal ||
     this->param.preconditioner_linearized_navier_stokes ==
       PreconditionerLinearizedNavierStokes::BlockTriangular ||
     this->param.preconditioner_linearized_navier_stokes ==
       PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
  {
    BlockPreconditionerData preconditioner_data;
    // clang-format off
    preconditioner_data.preconditioner_type = this->param.preconditioner_linearized_navier_stokes;
    preconditioner_data.momentum_preconditioner = this->param.momentum_preconditioner;
    preconditioner_data.exact_inversion_of_momentum_block = this->param.exact_inversion_of_momentum_block;
    preconditioner_data.multigrid_data_momentum_preconditioner = this->param.multigrid_data_momentum_preconditioner;
    preconditioner_data.rel_tol_solver_momentum_preconditioner = this->param.rel_tol_solver_momentum_preconditioner;
    preconditioner_data.max_n_tmp_vectors_solver_momentum_preconditioner = this->param.max_n_tmp_vectors_solver_momentum_preconditioner;
    preconditioner_data.schur_complement_preconditioner = this->param.schur_complement_preconditioner;
    preconditioner_data.discretization_of_laplacian = this->param.discretization_of_laplacian;
    preconditioner_data.exact_inversion_of_laplace_operator = this->param.exact_inversion_of_laplace_operator;
    preconditioner_data.multigrid_data_schur_complement_preconditioner = this->param.multigrid_data_schur_complement_preconditioner;
    preconditioner_data.rel_tol_solver_schur_complement_preconditioner = this->param.rel_tol_solver_schur_complement_preconditioner;
    // clang-format on

    preconditioner.reset(new Preconditioner(this, preconditioner_data));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::initialize_solver_coupled()
{
  // setup linear solver
  if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter              = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs  = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel  = this->param.rel_tol_linear;
    solver_data.right_preconditioning = this->param.use_right_preconditioning;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors     = this->param.max_n_tmp_vectors;
    solver_data.compute_eigenvalues   = false;

    if(this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(
      new GMRESSolver<THIS, Preconditioner, BlockVectorType>(*this, *preconditioner, solver_data));
  }
  else if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter              = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs  = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel  = this->param.rel_tol_linear;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors     = this->param.max_n_tmp_vectors;

    if(this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes ==
         PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(
      new FGMRESSolver<THIS, Preconditioner, BlockVectorType>(*this, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_linearized_navier_stokes ==
                    SolverLinearizedNavierStokes::GMRES ||
                  this->param.solver_linearized_navier_stokes ==
                    SolverLinearizedNavierStokes::FGMRES,
                ExcMessage("Specified solver for linearized Navier-Stokes problem not available."));
  }

  // setup Newton solver
  if(nonlinear_problem_has_to_be_solved())
  {
    newton_solver.reset(
      new NewtonSolver<BlockVectorType, THIS, THIS, IterativeSolverBase<BlockVectorType>>(
        this->param.newton_solver_data_coupled, *this, *this, *linear_solver));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::update_divergence_penalty_operator(
  VectorType const & velocity) const
{
  this->projection_operator->calculate_array_div_penalty_parameter(velocity);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::update_continuity_penalty_operator(
  VectorType const & velocity) const
{
  this->projection_operator->calculate_array_conti_penalty_parameter(velocity);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::initialize_block_vector_velocity_pressure(
  BlockVectorType & src) const
{
  // velocity(1st block) + pressure(2nd block)
  src.reinit(2);

  this->data.initialize_dof_vector(src.block(0), this->get_dof_index_velocity());
  this->data.initialize_dof_vector(src.block(1), this->get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::initialize_vector_for_newton_solver(
  BlockVectorType & src) const
{
  initialize_block_vector_velocity_pressure(src);
}

template<int dim, int degree_u, int degree_p, typename Number>
bool
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::nonlinear_problem_has_to_be_solved() const
{
  return this->param.equation_type == EquationType::NavierStokes &&
         (this->param.solver_type == SolverType::Steady ||
          (this->param.solver_type == SolverType::Unsteady &&
           this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit));
}

template<int dim, int degree_u, int degree_p, typename Number>
bool
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::unsteady_problem_has_to_be_solved() const
{
  return (this->param.solver_type == SolverType::Unsteady);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::set_scaling_factor_continuity(
  double const scaling_factor)
{
  scaling_factor_continuity = scaling_factor;
  this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::set_sum_alphai_ui(VectorType const * vector)
{
  this->sum_alphai_ui = vector;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::set_solution_linearization(
  BlockVectorType const & solution_linearization)
{
  momentum_operator.set_solution_linearization(solution_linearization.block(0));
}

template<int dim, int degree_u, int degree_p, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::get_velocity_linearization() const
{
  AssertThrow(nonlinear_problem_has_to_be_solved() == true,
              ExcMessage(
                "Attempt to access velocity_linearization which has not been initialized."));

  return momentum_operator.get_solution_linearization();
}

template<int dim, int degree_u, int degree_p, typename Number>
CompatibleLaplaceOperatorData<dim> const
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::get_compatible_laplace_operator_data() const
{
  CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data;
  comp_laplace_operator_data.dof_index_velocity       = this->get_dof_index_velocity();
  comp_laplace_operator_data.dof_index_pressure       = this->get_dof_index_pressure();
  comp_laplace_operator_data.dof_handler_u            = &this->get_dof_handler_u();
  comp_laplace_operator_data.gradient_operator_data   = this->get_gradient_operator_data();
  comp_laplace_operator_data.divergence_operator_data = this->get_divergence_operator_data();
  comp_laplace_operator_data.underlying_operator_dof_index_velocity =
    this->get_dof_index_velocity();

  return comp_laplace_operator_data;
}

template<int dim, int degree_u, int degree_p, typename Number>
unsigned int
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::solve_linear_stokes_problem(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  double const &          scaling_factor_mass_matrix_term)
{
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  momentum_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the velocity_conv_diff_operator
  // because this function is only called if the convective term is not considered
  // in the velocity_conv_diff_operator (Stokes eq. or explicit treatment of convective term).

  return linear_solver->solve(dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::rhs_stokes_problem(
  BlockVectorType & dst,
  double const &    eval_time) const
{
  // velocity-block
  this->gradient_operator.rhs(dst.block(0), eval_time);
  dst.block(0) *= scaling_factor_continuity;

  this->viscous_operator.rhs_add(dst.block(0), eval_time);

  if(this->param.right_hand_side == true)
    this->body_force_operator.evaluate_add(dst.block(0), eval_time);

  // Divergence and continuity penalty operators: no contribution to rhs

  // pressure-block
  this->divergence_operator.rhs(dst.block(1), eval_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::vmult(BlockVectorType &       dst,
                                                              BlockVectorType const & src) const
{
  // (1,1) block of saddle point matrix
  momentum_operator.vmult(dst.block(0), src.block(0));

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->projection_operator->apply_add_div_penalty(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->projection_operator->apply_add_conti_penalty(dst.block(0), src.block(0));
  }

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply(temp_vector, src.block(1));
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1), src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::solve_nonlinear_steady_problem(
  BlockVectorType & dst,
  unsigned int &    newton_iterations,
  unsigned int &    linear_iterations)
{
  // solve nonlinear problem
  newton_solver->solve(dst, newton_iterations, linear_iterations);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::solve_nonlinear_problem(
  BlockVectorType &  dst,
  VectorType const & sum_alphai_ui,
  double const &     eval_time,
  double const &     scaling_factor_mass_matrix_term,
  unsigned int &     newton_iterations,
  unsigned int &     linear_iterations)
{
  // Set sum_alphai_ui (this variable is used when evaluating the nonlinear residual).
  this->sum_alphai_ui = &sum_alphai_ui;

  // Set evaluation_time for nonlinear operator (=DGNavierStokesCoupled)
  evaluation_time = eval_time;
  // Set scaling_factor_time_derivative_term for nonlinear operator (=DGNavierStokesCoupled)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // Set correct evaluation time for linear operator (velocity_conv_diff_operator).
  momentum_operator.set_evaluation_time(eval_time);
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  momentum_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  newton_solver->solve(dst, newton_iterations, linear_iterations);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::evaluate_nonlinear_residual(
  BlockVectorType &       dst,
  BlockVectorType const & src) const
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0), evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(unsteady_problem_has_to_be_solved())
  {
    temp_vector.equ(scaling_factor_time_derivative_term, src.block(0));
    temp_vector.add(-1.0, *sum_alphai_ui);
    this->mass_matrix_operator.apply_add(dst.block(0), temp_vector);
  }

  this->convective_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);
  this->viscous_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->projection_operator->apply_add_div_penalty(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->projection_operator->apply_add_conti_penalty(dst.block(0), src.block(0));
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), evaluation_time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::evaluate_nonlinear_residual_steady(
  BlockVectorType &       dst,
  BlockVectorType const & src) const
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0), evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(this->param.equation_type == EquationType::NavierStokes)
    this->convective_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);

  this->viscous_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->projection_operator->apply_add_div_penalty(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->projection_operator->apply_add_conti_penalty(dst.block(0), src.block(0));
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), evaluation_time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::do_postprocessing(
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
DGNavierStokesCoupled<dim, degree_u, degree_p, Number>::do_postprocessing_steady_problem(
  VectorType const & velocity,
  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity,
                                         velocity, // intermediate_velocity
                                         pressure);
}

} // namespace IncNS

#include "dg_navier_stokes_coupled_solver.hpp"
