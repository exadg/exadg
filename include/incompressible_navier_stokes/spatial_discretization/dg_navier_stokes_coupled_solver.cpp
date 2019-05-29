/*
 * dg_navier_stokes_coupled_solver.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_coupled_solver.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesCoupled<dim, Number>::DGNavierStokesCoupled(
  parallel::Triangulation<dim> const & triangulation,
  InputParameters const &              parameters_in,
  std::shared_ptr<Postprocessor>       postprocessor_in)
  : BASE(triangulation, parameters_in, postprocessor_in),
    sum_alphai_ui(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0),
    scaling_factor_continuity(1.0)
{
}

template<int dim, typename Number>
DGNavierStokesCoupled<dim, Number>::~DGNavierStokesCoupled()
{
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup(
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in)
{
  BASE::setup(periodic_face_pairs,
              boundary_descriptor_velocity_in,
              boundary_descriptor_pressure_in,
              field_functions_in);

  this->initialize_vector_velocity(temp_vector);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_solvers(
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

  initialize_block_preconditioner();

  initialize_solver_coupled();

  this->setup_projection_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_momentum_operator(
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
  if(this->param.nonlinear_problem_has_to_be_solved())
    momentum_operator_data.convective_problem = true;
  else
    momentum_operator_data.convective_problem = false;

  momentum_operator_data.dof_index       = this->get_dof_index_velocity();
  momentum_operator_data.quad_index_std  = this->get_quad_index_velocity_linear();
  momentum_operator_data.quad_index_over = this->get_quad_index_velocity_nonlinear();

  momentum_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  momentum_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;
  momentum_operator_data.mg_operator_type = this->param.multigrid_operator_type_velocity_block;

  momentum_operator.reinit(this->get_matrix_free(),
                           momentum_operator_data,
                           this->mass_matrix_operator,
                           this->viscous_operator,
                           this->convective_operator);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_solver_coupled()
{
  // setup linear solver
  if(this->param.solver_coupled == SolverCoupled::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;
    solver_data.compute_eigenvalues  = false;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new GMRESSolver<THIS, Preconditioner, BlockVectorType>(*this,
                                                                               block_preconditioner,
                                                                               solver_data));
  }
  else if(this->param.solver_coupled == SolverCoupled::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new FGMRESSolver<THIS, Preconditioner, BlockVectorType>(
      *this, block_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for linearized problem is not implemented."));
  }

  // setup Newton solver
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    newton_solver.reset(
      new NewtonSolver<BlockVectorType, THIS, THIS, IterativeSolverBase<BlockVectorType>>(
        this->param.newton_solver_data_coupled, *this, *this, *linear_solver));
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::update_divergence_penalty_operator(
  VectorType const & velocity) const
{
  this->projection_operator->calculate_div_penalty_parameter(velocity);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::update_continuity_penalty_operator(
  VectorType const & velocity) const
{
  this->projection_operator->calculate_conti_penalty_parameter(velocity);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_block_vector_velocity_pressure(
  BlockVectorType & src) const
{
  // velocity(1st block) + pressure(2nd block)
  src.reinit(2);

  this->matrix_free.initialize_dof_vector(src.block(0), this->get_dof_index_velocity());
  this->matrix_free.initialize_dof_vector(src.block(1), this->get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_vector_for_newton_solver(BlockVectorType & src) const
{
  initialize_block_vector_velocity_pressure(src);
}

template<int dim, typename Number>
bool
DGNavierStokesCoupled<dim, Number>::nonlinear_problem_has_to_be_solved() const
{
  return this->param.nonlinear_problem_has_to_be_solved();
}

template<int dim, typename Number>
bool
DGNavierStokesCoupled<dim, Number>::unsteady_problem_has_to_be_solved() const
{
  return (this->param.solver_type == SolverType::Unsteady);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::set_scaling_factor_continuity(double const scaling_factor)
{
  scaling_factor_continuity = scaling_factor;
  this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::set_sum_alphai_ui(VectorType const * vector)
{
  this->sum_alphai_ui = vector;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::set_solution_linearization(
  BlockVectorType const & solution_linearization)
{
  momentum_operator.set_solution_linearization(solution_linearization.block(0));
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
DGNavierStokesCoupled<dim, Number>::get_velocity_linearization() const
{
  AssertThrow(this->param.nonlinear_problem_has_to_be_solved() == true,
              ExcMessage(
                "Attempt to access velocity_linearization which has not been initialized."));

  return momentum_operator.get_solution_linearization();
}

template<int dim, typename Number>
unsigned int
DGNavierStokesCoupled<dim, Number>::solve_linear_stokes_problem(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  bool const &            update_preconditioner,
  double const &          scaling_factor_mass_matrix_term)
{
  // Set scaling_factor_time_derivative_term for linear operator.
  momentum_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the momentum_operator
  // because this function is only called if the convective term is not considered
  // in the momentum_operator (Stokes eq. or explicit treatment of convective term).

  return linear_solver->solve(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::rhs_stokes_problem(BlockVectorType & dst,
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

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src) const
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

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::solve_nonlinear_steady_problem(
  BlockVectorType & dst,
  bool const &      update_preconditioner,
  unsigned int &    newton_iterations,
  unsigned int &    linear_iterations)
{
  // solve nonlinear problem
  newton_solver->solve(dst,
                       newton_iterations,
                       linear_iterations,
                       update_preconditioner,
                       this->param.update_preconditioner_coupled_every_newton_iter);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::solve_nonlinear_problem(
  BlockVectorType &  dst,
  VectorType const & sum_alphai_ui,
  double const &     eval_time,
  bool const &       update_preconditioner,
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
  newton_solver->solve(dst,
                       newton_iterations,
                       linear_iterations,
                       update_preconditioner,
                       this->param.update_preconditioner_coupled_every_newton_iter);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::evaluate_nonlinear_residual(BlockVectorType &       dst,
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

  AssertThrow(this->param.convective_problem() == true, ExcMessage("Invalid parameters."));

  this->convective_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);

  if(this->param.viscous_problem())
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

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::evaluate_nonlinear_residual_steady(
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

  if(this->param.convective_problem())
    this->convective_operator.evaluate_add(dst.block(0), src.block(0), evaluation_time);

  if(this->param.viscous_problem())
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

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::do_postprocessing(VectorType const & velocity,
                                                      VectorType const & pressure,
                                                      double const       time,
                                                      unsigned int const time_step_number) const
{
  bool const standard = true;
  if(standard)
  {
    this->postprocessor->do_postprocessing(velocity, pressure, time, time_step_number);
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
                                           pressure_error, // error!
                                           time,
                                           time_step_number);
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::do_postprocessing_steady_problem(
  VectorType const & velocity,
  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity, pressure);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_block_preconditioner()
{
  block_preconditioner.initialize(this);

  initialize_vectors();

  initialize_preconditioner_velocity_block();

  initialize_preconditioner_pressure_block();
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_vectors()
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockTriangular)
  {
    this->initialize_vector_velocity(vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    this->initialize_vector_pressure(vec_tmp_pressure);
    this->initialize_vector_velocity(vec_tmp_velocity);
    this->initialize_vector_velocity(vec_tmp_velocity_2);
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_preconditioner_velocity_block()
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::PointJacobi)
  {
    // Point Jacobi preconditioner
    preconditioner_momentum.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(momentum_operator));
  }
  else if(type == MomentumPreconditioner::BlockJacobi)
  {
    // Block Jacobi preconditioner
    preconditioner_momentum.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(momentum_operator));
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    // inverse mass matrix
    preconditioner_momentum.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
      this->get_matrix_free(),
      this->param.degree_u,
      this->get_dof_index_velocity(),
      this->get_quad_index_velocity_linear()));
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    setup_multigrid_preconditioner_momentum();

    if(this->param.exact_inversion_of_velocity_block == true)
    {
      setup_iterative_solver_momentum();
    }
  }
  else
  {
    AssertThrow(type == MomentumPreconditioner::None, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_multigrid_preconditioner_momentum()
{
  typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

  preconditioner_momentum.reset(new MULTIGRID());

  std::shared_ptr<MULTIGRID> mg_preconditioner =
    std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

  auto & dof_handler = this->get_dof_handler_u();

  parallel::Triangulation<dim> const * tria =
    dynamic_cast<parallel::Triangulation<dim> const *>(&dof_handler.get_triangulation());
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  mg_preconditioner->initialize(this->param.multigrid_data_velocity_block,
                                tria,
                                fe,
                                this->get_mapping(),
                                momentum_operator.get_operator_data());
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_iterative_solver_momentum()
{
  AssertThrow(preconditioner_momentum.get() != 0,
              ExcMessage("preconditioner_momentum is uninitialized"));

  // use FMGRES for "exact" solution of velocity block system
  FGMRESSolverData gmres_data;
  gmres_data.use_preconditioner   = true;
  gmres_data.max_iter             = this->param.solver_data_velocity_block.max_iter;
  gmres_data.solver_tolerance_abs = this->param.solver_data_velocity_block.abs_tol;
  gmres_data.solver_tolerance_rel = this->param.solver_data_velocity_block.rel_tol;
  gmres_data.max_n_tmp_vectors    = this->param.solver_data_velocity_block.max_krylov_size;

  solver_velocity_block.reset(
    new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
      momentum_operator, *preconditioner_momentum, gmres_data));
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_preconditioner_pressure_block()
{
  auto type = this->param.preconditioner_pressure_block;

  if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                          this->param.get_degree_p(),
                                                          this->get_dof_index_pressure(),
                                                          this->get_quad_index_pressure()));
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    AssertThrow(this->unsteady_problem_has_to_be_solved() == true,
                ExcMessage(
                  "Cahouet-Chabard preconditioner only makes sense for unsteady problems."));

    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // inverse mass matrix to also include the part of the preconditioner that is beneficial when
    // using large time steps and large viscosities.
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                          this->param.get_degree_p(),
                                                          this->get_dof_index_pressure(),
                                                          this->get_quad_index_pressure()));

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::Elman)
  {
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    if(this->param.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
      // --> inverse velocity mass matrix needed for inner factor
      inv_mass_matrix_preconditioner_schur_complement.reset(
        new InverseMassMatrixPreconditioner<dim, dim, Number>(
          this->get_matrix_free(),
          this->param.degree_u,
          this->get_dof_index_velocity(),
          this->get_quad_index_velocity_linear()));
    }

    // initialize tmp vectors
    this->initialize_vector_pressure(tmp_scp_pressure);
    this->initialize_vector_velocity(tmp_scp_velocity);
    this->initialize_vector_velocity(tmp_scp_velocity_2);
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // II. pressure convection-diffusion operator
    setup_pressure_convection_diffusion_operator();

    // III. inverse pressure mass matrix
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                          this->param.get_degree_p(),
                                                          this->get_dof_index_pressure(),
                                                          this->get_quad_index_pressure()));

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else
  {
    AssertThrow(type == SchurComplementPreconditioner::None, ExcNotImplemented());
  }
}

template<int dim, typename Number>
CompatibleLaplaceOperatorData<dim> const
DGNavierStokesCoupled<dim, Number>::get_compatible_laplace_operator_data() const
{
  CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data;
  comp_laplace_operator_data.degree_u                 = this->param.degree_u;
  comp_laplace_operator_data.degree_p                 = this->param.get_degree_p();
  comp_laplace_operator_data.dof_index_velocity       = this->get_dof_index_velocity();
  comp_laplace_operator_data.dof_index_pressure       = this->get_dof_index_pressure();
  comp_laplace_operator_data.operator_is_singular     = this->param.pure_dirichlet_bc;
  comp_laplace_operator_data.dof_handler_u            = &this->get_dof_handler_u();
  comp_laplace_operator_data.gradient_operator_data   = this->get_gradient_operator_data();
  comp_laplace_operator_data.divergence_operator_data = this->get_divergence_operator_data();

  return comp_laplace_operator_data;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_multigrid_preconditioner_schur_complement()
{
  auto type_laplacian = this->param.discretization_of_laplacian;

  if(type_laplacian == DiscretizationOfLaplacian::Compatible)
  {
    MultigridData mg_data = this->param.multigrid_data_pressure_block;

    typedef CompatibleLaplaceMultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    multigrid_preconditioner_schur_complement.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(multigrid_preconditioner_schur_complement);

    auto compatible_laplace_operator_data = get_compatible_laplace_operator_data();

    auto & dof_handler = this->get_dof_handler_p();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(
      mg_data, tria, fe, this->get_mapping(), compatible_laplace_operator_data);
  }
  else if(type_laplacian == DiscretizationOfLaplacian::Classical)
  {
    // multigrid V-cycle for negative Laplace operator
    Poisson::LaplaceOperatorData<dim> laplace_operator_data;
    laplace_operator_data.dof_index            = this->get_dof_index_pressure();
    laplace_operator_data.quad_index           = this->get_quad_index_pressure();
    laplace_operator_data.IP_factor            = 1.0;
    laplace_operator_data.degree               = this->param.get_degree_p();
    laplace_operator_data.degree_mapping       = this->mapping_degree;
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;

    laplace_operator_data.bc = this->boundary_descriptor_laplace;

    MultigridData mg_data = this->param.multigrid_data_pressure_block;

    typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    multigrid_preconditioner_schur_complement.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(multigrid_preconditioner_schur_complement);

    auto & dof_handler = this->get_dof_handler_p();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  laplace_operator_data,
                                  &laplace_operator_data.bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
  }
  else
  {
    AssertThrow(
      type_laplacian == DiscretizationOfLaplacian::Classical ||
        type_laplacian == DiscretizationOfLaplacian::Compatible,
      ExcMessage(
        "Specified discretization of Laplacian for Schur-complement preconditioner is not available."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_iterative_solver_schur_complement()
{
  AssertThrow(
    multigrid_preconditioner_schur_complement.get() != 0,
    ExcMessage(
      "Setup of iterative solver for Schur complement preconditioner: Multigrid preconditioner is uninitialized"));

  CGSolverData solver_data;
  solver_data.max_iter             = this->param.solver_data_pressure_block.max_iter;
  solver_data.solver_tolerance_abs = this->param.solver_data_pressure_block.abs_tol;
  solver_data.solver_tolerance_rel = this->param.solver_data_pressure_block.rel_tol;
  solver_data.use_preconditioner   = true;

  auto type_laplacian = this->param.discretization_of_laplacian;

  if(type_laplacian == DiscretizationOfLaplacian::Classical)
  {
    Poisson::LaplaceOperatorData<dim> laplace_operator_data;
    laplace_operator_data.dof_index      = this->get_dof_index_pressure();
    laplace_operator_data.quad_index     = this->get_quad_index_pressure();
    laplace_operator_data.IP_factor      = 1.0;
    laplace_operator_data.degree         = this->param.get_degree_p();
    laplace_operator_data.degree_mapping = this->mapping_degree;
    laplace_operator_data.bc             = this->boundary_descriptor_laplace;

    laplace_operator_classical.reset(new Poisson::LaplaceOperator<dim, Number>());
    laplace_operator_classical->reinit(this->get_matrix_free(),
                                       this->constraint_p,
                                       laplace_operator_data);

    solver_pressure_block.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        *laplace_operator_classical, *multigrid_preconditioner_schur_complement, solver_data));
  }
  else if(type_laplacian == DiscretizationOfLaplacian::Compatible)
  {
    CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data =
      get_compatible_laplace_operator_data();

    laplace_operator_compatible.reset(new CompatibleLaplaceOperator<dim, Number>());

    laplace_operator_compatible->initialize(this->get_matrix_free(),
                                            compatible_laplace_operator_data,
                                            this->gradient_operator,
                                            this->divergence_operator,
                                            this->inverse_mass_velocity);

    solver_pressure_block.reset(
      new CGSolver<CompatibleLaplaceOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        *laplace_operator_compatible, *multigrid_preconditioner_schur_complement, solver_data));
  }
  else
  {
    AssertThrow(
      type_laplacian == DiscretizationOfLaplacian::Classical ||
        type_laplacian == DiscretizationOfLaplacian::Compatible,
      ExcMessage(
        "Specified discretization of Laplacian for Schur-complement preconditioner is not available."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_pressure_convection_diffusion_operator()
{
  // pressure convection-diffusion operator
  // a) mass matrix operator
  ConvDiff::MassMatrixOperatorData mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index  = this->get_dof_index_pressure();
  mass_matrix_operator_data.quad_index = this->get_quad_index_pressure();

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<dim>());

  // for the pressure convection-diffusion operator the homogeneous operators (convective,
  // diffusive) are applied, so there is no need to specify functions for boundary conditions
  // since they will not be used (must not be used)
  // -> use ConstantFunction as dummy, initialized with NAN in order to detect a possible
  // incorrect access to boundary values
  std::shared_ptr<Function<dim>> dummy;

  // set boundary ID's for pressure convection-diffusion operator

  // Dirichlet BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        this->boundary_descriptor_pressure->dirichlet_bc.begin();
      it != this->boundary_descriptor_pressure->dirichlet_bc.end();
      ++it)
  {
    boundary_descriptor->dirichlet_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }
  // Neumann BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        this->boundary_descriptor_pressure->neumann_bc.begin();
      it != this->boundary_descriptor_pressure->neumann_bc.end();
      ++it)
  {
    boundary_descriptor->neumann_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }

  // b) diffusive operator
  ConvDiff::DiffusiveOperatorData<dim> diffusive_operator_data;
  diffusive_operator_data.dof_index      = this->get_dof_index_pressure();
  diffusive_operator_data.quad_index     = this->get_quad_index_pressure();
  diffusive_operator_data.IP_factor      = this->param.IP_factor_viscous;
  diffusive_operator_data.degree         = this->param.get_degree_p();
  diffusive_operator_data.degree_mapping = this->mapping_degree;
  diffusive_operator_data.bc             = boundary_descriptor;
  // TODO: the pressure convection-diffusion operator is initialized with constant viscosity, in
  // case of varying viscosities the pressure convection-diffusion operator (the diffusive
  // operator of the pressure convection-diffusion operator) has to be updated before applying
  // this preconditioner
  diffusive_operator_data.diffusivity = this->get_viscosity();

  // c) convective operator
  ConvDiff::ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index           = this->get_dof_index_pressure();
  convective_operator_data.dof_index_velocity  = this->get_dof_index_velocity();
  convective_operator_data.quad_index          = this->get_quad_index_velocity_linear();
  convective_operator_data.type_velocity_field = ConvDiff::TypeVelocityField::Numerical;
  convective_operator_data.numerical_flux_formulation =
    ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux;
  convective_operator_data.bc = boundary_descriptor;

  PressureConvectionDiffusionOperatorData<dim> pressure_convection_diffusion_operator_data;
  pressure_convection_diffusion_operator_data.mass_matrix_operator_data = mass_matrix_operator_data;
  pressure_convection_diffusion_operator_data.diffusive_operator_data   = diffusive_operator_data;
  pressure_convection_diffusion_operator_data.convective_operator_data  = convective_operator_data;
  if(unsteady_problem_has_to_be_solved())
    pressure_convection_diffusion_operator_data.unsteady_problem = true;
  else
    pressure_convection_diffusion_operator_data.unsteady_problem = false;
  pressure_convection_diffusion_operator_data.convective_problem =
    nonlinear_problem_has_to_be_solved();

  pressure_convection_diffusion_operator.reset(new PressureConvectionDiffusionOperator<dim, Number>(
    *this->mapping,
    this->get_matrix_free(),
    pressure_convection_diffusion_operator_data,
    this->constraint_p));

  if(unsteady_problem_has_to_be_solved())
    pressure_convection_diffusion_operator->set_scaling_factor_time_derivative_term(
      this->momentum_operator.get_scaling_factor_time_derivative_term());
}

// clang-format off
/*
 * Consider the following saddle point matrix :
 *
 *       / A  B^{T} \
 *   M = |          |
 *       \ B    0   /
 *
 *  with block factorization
 *
 *       / I         0 \  / A   0 \ / I   A^{-1} B^{T} \
 *   M = |             |  |       | |                  |
 *       \ B A^{-1}  I /  \ 0   S / \ 0        I       /
 *
 *       / I         0 \  / A   B^{T} \
 *     = |             |  |           |
 *       \ B A^{-1}  I /  \ 0     S   /
 *
 *        / A  0 \  / I   A^{-1} B^{T} \
 *     =  |      |  |                  |
 *        \ B  S /  \ 0        I       /
 *
 *   with Schur complement S = -B A^{-1} B^{T}
 *
 *
 * - Block-diagonal preconditioner:
 *
 *                   / A   0 \                       / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
 *   -> P_diagonal = |       |  -> P_diagonal^{-1} = |               | = |           | * |             |
 *                   \ 0  -S /                       \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
 *
 * - Block-triangular preconditioner:
 *
 *                     / A   B^{T} \                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
 *   -> P_triangular = |           |  -> P_triangular^{-1} = |           | * |          | * |             |
 *                     \ 0     S   /                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
 *
 * - Block-triangular factorization:
 *
 *                      / A  0 \  / I   A^{-1} B^{T} \
 *   -> P_tria-factor = |      |  |                  |
 *                      \ B  S /  \ 0        I       /
 *
 *                            / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1}  0 \
 *    -> P_tria-factor^{-1} = |                   | * |             | * |       | * |           |
 *                            \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0     I /
 *
 *
 *  Main challenge: Development of efficient preconditioners for A and S that approximate
 *  the velocity block A and the Schur-complement block S in a spectrally equivalent way.
 *
 *
 *  Approximations of velocity block A = 1/dt M_u + C_lin(u) + nu (-L):
 *
 *   1. inverse mass matrix preconditioner (dt small):
 *
 *     A = 1/dt M_u
 *
 *     -> A^{-1} = dt M_u^{-1}
 *
 *   2. Helmholtz operator H =  1/dt M_u + nu (-L) (neglecting the convective term):
 *
 *     -> A^{-1} = H^{-1} where H^{-1} is approximated by performing one multigrid V-cycle for the Helmholtz operator
 *
 *   3. Velocity convection-diffusion operator A = 1/dt M_u + C_lin(u) + nu (-L) including the convective term:
 *
 *      -> to approximately invert A consider one multigrid V-cycle
 *
 *  Approximations of pressure Schur-complement block S:
 *
 *   - S = - B A^{-1} B^T
 *       |
 *       |  apply method of pseudo-differential operators and neglect convective term
 *      \|/
 *       = - (- div ) * ( 1/dt * I - nu * laplace )^{-1} * grad
 *
 *   1. dt small, nu small:

 *      S = div * (1/dt I)^{-1} * grad = dt * laplace
 *
 *      -> - S^{-1} = 1/dt (-L)^{-1} (-L: negative Laplace operator)
 *
 *   2. dt large, nu large:
 *
 *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I
 *
 *      -> - S^{-1} = nu M_p^{-1} (M_p: pressure mass matrix)
 *
 *   3. Cahouet & Chabard (combines 1. and 2., robust preconditioner for whole range of time step sizes and visosities)
 *
 *      -> - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}
 *
 *   4. Elman et al. (BFBt preconditioner, sparse approximate commutator preconditioner)
 *
 *      S = - B A^{-1}B^T is approximated by (BB^T) (-B A B^T)^{-1} (BB^T)
 *
 *      -> -S^{-1} = - (-L)^{-1} (-B A B^T) (-L)^{-1}
 *
 *      improvement: S is approximated by (BM^{-1}B^T) (-B A B^T)^{-1} (BM^{-1}B^T)
 *
 *      -> -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
 *
 *   5. Pressure convection-diffusion preconditioner
 *
 *      -> -S^{-1} = M_p^{-1} A_p (-L)^{-1} where A_p is a convection-diffusion operator for the pressure
 */
// clang-format on

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::update_block_preconditioner(THIS const * /*operator*/)
{
  // momentum block
  preconditioner_momentum->update(&momentum_operator);

  // pressure block
  if(this->param.preconditioner_pressure_block ==
     SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    if(unsteady_problem_has_to_be_solved())
    {
      pressure_convection_diffusion_operator->set_scaling_factor_time_derivative_term(
        momentum_operator.get_scaling_factor_time_derivative_term());
    }
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::apply_block_preconditioner(BlockVectorType &       dst,
                                                               BlockVectorType const & src) const
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockDiagonal)
  {
    /*                        / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
     *   -> P_diagonal^{-1} = |               | = |           | * |             |
     *                        \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
     */

    /*
     *         / I      0    \
     *  temp = |             | * src
     *         \ 0   -S^{-1} /
     */

    // apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * temp
     *        \   0     I /
     */

    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));
  }
  else if(type == PreconditionerCoupled::BlockTriangular)
  {
    /*
     *                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *  -> P_triangular^{-1} = |           | * |          | * |             |
     *                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     */

    /*
     *        / I      0    \
     *  dst = |             | * src
     *        \ 0   -S^{-1} /
     */

    // For the velocity block simply copy data from src to dst.
    dst.block(0) = src.block(0);
    // Apply preconditioner for pressure/Schur-complement block.
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / I  B^{T} \
     *  dst = |          | * dst
     *        \ 0   -I   /
     */

    // Apply gradient operator and add to dst vector.
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));
    dst.block(0).add(this->scaling_factor_continuity, vec_tmp_velocity);
    dst.block(1) *= -1.0;

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * dst
     *        \   0     I /
     */

    // Copy data from dst.block(0) to vec_tmp_velocity before
    // applying the preconditioner for the velocity block.
    vec_tmp_velocity = dst.block(0);
    // Apply preconditioner for velocity/momentum block.
    apply_preconditioner_velocity_block(dst.block(0), vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    /*
     *                          / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1} 0 \
     *  -> P_tria-factor^{-1} = |                   | * |             | * |       | * |          |
     *                          \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0    I /
     */

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * src
     *        \   0     I /
     */

    // for the pressure block simply copy data from src to dst
    dst.block(1) = src.block(1);
    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));

    /*
     *        / I   0 \
     *  dst = |       | * dst
     *        \ B  -I /
     */

    // dst.block(1) = B*dst.block(0) - dst.block(1)
    //              = -1.0 * (dst.block(1) + (-B) * dst.block(0));
    // I. dst.block(1) += (-B) * dst.block(0);
    // Note that B represents NEGATIVE divergence operator, i.e.,
    // applying -B is equal to applying the divergence operator
    this->divergence_operator.apply(vec_tmp_pressure, dst.block(0));
    dst.block(1).add(this->scaling_factor_continuity, vec_tmp_pressure);
    // II. dst.block(1) = -dst.block(1);
    dst.block(1) *= -1.0;

    /*
     *        / I      0    \
     *  dst = |             | * dst
     *        \ 0   -S^{-1} /
     */

    // Copy data from dst.block(1) to vec_tmp_pressure before
    // applying the preconditioner for the pressure block.
    vec_tmp_pressure = dst.block(1);
    // Apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), vec_tmp_pressure);

    /*
     *        / I  - A^{-1} B^{T} \
     *  dst = |                   | * dst
     *        \ 0          I      /
     */

    // vec_tmp_velocity = B^{T} * dst(1)
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));

    // scaling factor continuity
    vec_tmp_velocity *= this->scaling_factor_continuity;

    // vec_tmp_velocity_2 = A^{-1} * vec_tmp_velocity
    apply_preconditioner_velocity_block(vec_tmp_velocity_2, vec_tmp_velocity);

    // dst(0) = dst(0) - vec_tmp_velocity_2
    dst.block(0).add(-1.0, vec_tmp_velocity_2);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::apply_preconditioner_velocity_block(
  VectorType &       dst,
  VectorType const & src) const
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::None)
  {
    dst = src;
  }
  else if(type == MomentumPreconditioner::PointJacobi ||
          type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum->vmult(dst, src);
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    // use the inverse mass matrix as an approximation to the momentum block
    preconditioner_momentum->vmult(dst, src);
    // clang-format off
    dst *= 1. / momentum_operator.get_scaling_factor_time_derivative_term();
    // clang-format on
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    if(this->param.exact_inversion_of_velocity_block == false)
    {
      // perform one multigrid V-cylce
      preconditioner_momentum->vmult(dst, src);
    }
    else // exact_inversion_of_velocity_block == true
    {
      // check correctness of multigrid V-cycle

      // clang-format off
      /*
      typedef MultigridPreconditioner<dim, degree_u, Number, MultigridNumber> MULTIGRID;

      std::shared_ptr<MULTIGRID> preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

      CheckMultigrid<dim, Number, MomentumOperator<dim, degree_u, Number>, MULTIGRID>
        check_multigrid(momentum_operator,preconditioner);

      check_multigrid.check();
      */
      // clang-format on

      // iteratively solve momentum equation up to given tolerance
      dst = 0.0;
      // Note that update of preconditioner is set to false here since the preconditioner has
      // already been updated in the member function update() if desired.
      unsigned int const iterations =
        solver_velocity_block->solve(dst, src, /* update_preconditioner = */ false);

      // output
      bool const print_iterations = false;
      if(print_iterations)
      {
        ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
        pcout << "Number of iterations for inner solver = " << iterations << std::endl;
      }
    }
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::apply_preconditioner_pressure_block(
  VectorType &       dst,
  VectorType const & src) const
{
  auto type = this->param.preconditioner_pressure_block;

  if(type == SchurComplementPreconditioner::None)
  {
    // No preconditioner for Schur-complement block
    dst = src;
  }
  else if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    // - S^{-1} = nu M_p^{-1}
    inv_mass_matrix_preconditioner_schur_complement->vmult(dst, src);
    dst *= this->get_viscosity();
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    // -S^{-1} = 1/dt  (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= momentum_operator.get_scaling_factor_time_derivative_term();
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

    // I. 1/dt (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= momentum_operator.get_scaling_factor_time_derivative_term();

    // II. M_p^{-1}, apply inverse pressure mass matrix to src-vector and store the result in a
    // temporary vector
    inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_pressure, src);

    // III. add temporary vector scaled by viscosity
    dst.add(this->get_viscosity(), tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::Elman)
  {
    auto type_laplacian = this->param.discretization_of_laplacian;

    if(type_laplacian == DiscretizationOfLaplacian::Classical)
    {
      // -S^{-1} = - (BB^T)^{-1} (-B A B^T) (BB^T)^{-1}

      // I. (BB^T)^{-1} -> apply inverse negative Laplace operator (classical discretization),
      // (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, src);

      // II. (-B A B^T)
      // II.a) B^T
      this->gradient_operator.apply(tmp_scp_velocity, dst);

      // II.b) A = 1/dt * mass matrix  +  viscous term  +  linearized convective term
      momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

      // II.c) -B
      this->divergence_operator.apply(tmp_scp_pressure, tmp_scp_velocity_2);

      // III. -(BB^T)^{-1}
      // III.a) apply inverse negative Laplace operator (classical discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, tmp_scp_pressure);
      // III.b) minus sign
      dst *= -1.0;
    }
    else if(type_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}

      // I. (BM^{-1}B^T)^{-1} -> apply inverse negative Laplace operator (compatible
      // discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, src);


      // II. (-B M^{-1} A M^{-1} B^T)
      // II.a) B^T
      this->gradient_operator.apply(tmp_scp_velocity, dst);

      // II.b) M^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity, tmp_scp_velocity);

      // II.c) A = 1/dt * mass matrix + viscous term + linearized convective term
      momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

      // II.d) M^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity_2,
                                                             tmp_scp_velocity_2);

      // II.e) -B
      this->divergence_operator.apply(tmp_scp_pressure, tmp_scp_velocity_2);


      // III. -(BM^{-1}B^T)^{-1}
      // III.a) apply inverse negative Laplace operator (compatible discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, tmp_scp_pressure);
      // III.b) minus sign
      dst *= -1.0;
    }
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. inverse, negative Laplace operator (-L)^{-1}
    apply_inverse_negative_laplace_operator(tmp_scp_pressure, src);

    // II. pressure convection diffusion operator A_p
    if(nonlinear_problem_has_to_be_solved() == true)
    {
      pressure_convection_diffusion_operator->apply(dst,
                                                    tmp_scp_pressure,
                                                    get_velocity_linearization());
    }
    else
    {
      VectorType dummy;
      pressure_convection_diffusion_operator->apply(dst, tmp_scp_pressure, dummy);
    }

    // III. inverse pressure mass matrix M_p^{-1}
    inv_mass_matrix_preconditioner_schur_complement->vmult(dst, dst);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  // scaling_factor_continuity: Since the Schur complement includes both the velocity divergence
  // and the pressure gradient operators as factors, we have to scale by
  // 1/(scaling_factor*scaling_factor) when applying (an approximation of) the inverse Schur
  // complement.
  double inverse_scaling_factor = 1.0 / scaling_factor_continuity;
  dst *= inverse_scaling_factor * inverse_scaling_factor;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::apply_inverse_negative_laplace_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  if(this->param.exact_inversion_of_laplace_operator == false)
  {
    // perform one multigrid V-cycle in order to approximately invert the negative Laplace
    // operator (classical or compatible)
    multigrid_preconditioner_schur_complement->vmult(dst, src);
  }
  else // exact_inversion_of_laplace_operator == true
  {
    // solve a linear system of equations for negative Laplace operator to given (relative)
    // tolerance using the PCG method
    VectorType const * pointer_to_src = &src;
    if(this->param.pure_dirichlet_bc == true)
    {
      VectorType vector_zero_mean;
      vector_zero_mean = src;

      auto type_laplacian = this->param.discretization_of_laplacian;

      bool singular = false;
      if(type_laplacian == DiscretizationOfLaplacian::Classical)
        singular = laplace_operator_classical->is_singular();
      else if(type_laplacian == DiscretizationOfLaplacian::Compatible)
        singular = laplace_operator_compatible->is_singular();
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      if(singular)
      {
        set_zero_mean_value(vector_zero_mean);
      }

      pointer_to_src = &vector_zero_mean;
    }

    dst = 0.0;
    // Note that update of preconditioner is set to false here since the preconditioner has
    // already been updated in the member function update() if desired.
    solver_pressure_block->solve(dst, *pointer_to_src, /* update_preconditioner = */ false);
  }
}


template class DGNavierStokesCoupled<2, float>;
template class DGNavierStokesCoupled<2, double>;

template class DGNavierStokesCoupled<3, float>;
template class DGNavierStokesCoupled<3, double>;

} // namespace IncNS
