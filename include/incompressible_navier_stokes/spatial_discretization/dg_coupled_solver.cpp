/*
 * dg_coupled_solver.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_coupled_solver.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesCoupled<dim, Number>::DGNavierStokesCoupled(
  parallel::TriangulationBase<dim> const & triangulation,
  InputParameters const &              parameters,
  std::shared_ptr<Postprocessor>       postprocessor)
  : Base(triangulation, parameters, postprocessor), scaling_factor_continuity(1.0)
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
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions)
{
  Base::setup(periodic_face_pairs,
              boundary_descriptor_velocity,
              boundary_descriptor_pressure,
              field_functions);

  this->initialize_vector_velocity(temp_vector);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::setup_solvers(
  double const &     scaling_factor_time_derivative_term,
  VectorType const & velocity)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  Base::setup_solvers(scaling_factor_time_derivative_term, velocity);

  initialize_block_preconditioner();

  initialize_solver_coupled();

  if(this->param.add_penalty_terms_to_monolithic_system == false)
    this->setup_projection_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::initialize_solver_coupled()
{
  linear_operator.initialize(*this);

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

    linear_solver.reset(
      new GMRESSolver<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>(
        linear_operator, block_preconditioner, solver_data));
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

    linear_solver.reset(
      new FGMRESSolver<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>(
        linear_operator, block_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for linearized problem is not implemented."));
  }

  // setup Newton solver
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    nonlinear_operator.initialize(*this);

    newton_solver.reset(new NewtonSolver<BlockVectorType,
                                         NonlinearOperatorCoupled<dim, Number>,
                                         LinearOperatorCoupled<dim, Number>,
                                         IterativeSolverBase<BlockVectorType>>(
      this->param.newton_solver_data_coupled, nonlinear_operator, linear_operator, *linear_solver));
  }
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::update_divergence_penalty_operator(VectorType const & velocity)
{
  this->div_penalty_operator.update(velocity);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::update_continuity_penalty_operator(VectorType const & velocity)
{
  this->conti_penalty_operator.update(velocity);
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
DGNavierStokesCoupled<dim, Number>::set_scaling_factor_continuity(double const scaling_factor)
{
  scaling_factor_continuity = scaling_factor;
  this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesCoupled<dim, Number>::solve_linear_stokes_problem(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  bool const &            update_preconditioner,
  double const &          time,
  double const &          scaling_factor_mass_matrix_term)
{
  // Update linear operator
  linear_operator.update(time, scaling_factor_mass_matrix_term);

  return linear_solver->solve(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::rhs_stokes_problem(BlockVectorType & dst,
                                                       double const &    time) const
{
  // velocity-block
  this->gradient_operator.rhs(dst.block(0), time);
  dst.block(0) *= scaling_factor_continuity;

  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(dst.block(0));

  if(this->param.right_hand_side == true)
    this->rhs_operator.evaluate_add(dst.block(0), time);

  // Divergence and continuity penalty operators: no contribution to rhs

  // pressure-block
  this->divergence_operator.rhs(dst.block(1), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::apply_linearized_problem(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  double const &          time,
  double const &          scaling_factor_mass_matrix) const
{
  // (1,1) block of saddle point matrix
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_matrix(scaling_factor_mass_matrix);
  this->momentum_operator.vmult(dst.block(0), src.block(0));

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.apply_add(dst.block(0), src.block(0));
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
  BlockVectorType &  dst,
  VectorType const & rhs_vector,
  bool const &       update_preconditioner,
  unsigned int &     newton_iterations,
  unsigned int &     linear_iterations)
{
  // update nonlinear operator
  nonlinear_operator.update(rhs_vector, 0.0 /* time */, 1.0 /* scaling_factor */);

  // no need to update linear operator since this is a steady problem

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
  VectorType const & rhs_vector,
  double const &     time,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass_matrix_term,
  unsigned int &     newton_iterations,
  unsigned int &     linear_iterations)
{
  // Update nonlinear operator
  nonlinear_operator.update(rhs_vector, time, scaling_factor_mass_matrix_term);

  // Update linear operator
  linear_operator.update(time, scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  newton_solver->solve(dst,
                       newton_iterations,
                       linear_iterations,
                       update_preconditioner,
                       this->param.update_preconditioner_coupled_every_newton_iter);
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::evaluate_nonlinear_residual(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  VectorType const *      rhs_vector,
  double const &          time,
  double const &          scaling_factor_mass_matrix) const
{
  // velocity-block

  if(this->unsteady_problem_has_to_be_solved())
    this->mass_matrix_operator.apply_scale(dst.block(0), scaling_factor_mass_matrix, src.block(0));
  else
    dst.block(0) = 0.0;

  AssertThrow(this->param.convective_problem() == true, ExcMessage("Invalid parameters."));

  this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.apply_add(dst.block(0), src.block(0));
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // constant right-hand side vector (body force vector and sum_alphai_ui term)
  dst.block(0).add(-1.0, *rhs_vector);

  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
DGNavierStokesCoupled<dim, Number>::evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                                                       BlockVectorType const & src,
                                                                       double const & time) const
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst.block(0), time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(this->param.convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);
  }

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.apply_add(dst.block(0), src.block(0));
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
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
    preconditioner_momentum.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
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

  parallel::TriangulationBase<dim> const * tria =
    dynamic_cast<parallel::TriangulationBase<dim> const *>(&dof_handler.get_triangulation());
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  mg_preconditioner->initialize(this->param.multigrid_data_velocity_block,
                                tria,
                                fe,
                                this->get_mapping(),
                                this->momentum_operator,
                                this->param.multigrid_operator_type_velocity_block);
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
      this->momentum_operator, *preconditioner_momentum, gmres_data));
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
  comp_laplace_operator_data.degree_u               = this->param.degree_u;
  comp_laplace_operator_data.degree_p               = this->param.get_degree_p();
  comp_laplace_operator_data.dof_index_velocity     = this->get_dof_index_velocity();
  comp_laplace_operator_data.dof_index_pressure     = this->get_dof_index_pressure();
  comp_laplace_operator_data.operator_is_singular   = this->param.pure_dirichlet_bc;
  comp_laplace_operator_data.dof_handler_u          = &this->get_dof_handler_u();
  comp_laplace_operator_data.gradient_operator_data = this->gradient_operator.get_operator_data();
  comp_laplace_operator_data.divergence_operator_data =
    this->divergence_operator.get_operator_data();

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

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(
      mg_data, tria, fe, this->get_mapping(), compatible_laplace_operator_data);
  }
  else if(type_laplacian == DiscretizationOfLaplacian::Classical)
  {
    // multigrid V-cycle for negative Laplace operator
    Poisson::LaplaceOperatorData<dim> laplace_operator_data;
    laplace_operator_data.dof_index                  = this->get_dof_index_pressure();
    laplace_operator_data.quad_index                 = this->get_quad_index_pressure();
    laplace_operator_data.operator_is_singular       = this->param.pure_dirichlet_bc;
    laplace_operator_data.kernel_data.IP_factor      = 1.0;
    laplace_operator_data.kernel_data.degree         = this->param.get_degree_p();
    laplace_operator_data.kernel_data.degree_mapping = this->mapping_degree;

    laplace_operator_data.bc = this->boundary_descriptor_laplace;

    MultigridData mg_data = this->param.multigrid_data_pressure_block;

    typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    multigrid_preconditioner_schur_complement.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(multigrid_preconditioner_schur_complement);

    auto & dof_handler = this->get_dof_handler_p();

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());
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
    laplace_operator_data.dof_index                  = this->get_dof_index_pressure();
    laplace_operator_data.quad_index                 = this->get_quad_index_pressure();
    laplace_operator_data.bc                         = this->boundary_descriptor_laplace;
    laplace_operator_data.kernel_data.IP_factor      = 1.0;
    laplace_operator_data.kernel_data.degree         = this->param.get_degree_p();
    laplace_operator_data.kernel_data.degree_mapping = this->mapping_degree;

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

  // fill boundary descriptor
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<dim>());

  // For the pressure convection-diffusion operator the homogeneous operators are applied, so there
  // is no need to specify functions for boundary conditions since they will never be used.
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

  // convective operator:
  // use numerical velocity field with dof index of velocity field and local Lax-Friedrichs flux to
  // mimic the upwind-like discretization of the linearized convective term in the Navier-Stokes
  // equations.
  ConvDiff::Operators::ConvectiveKernelData<dim> convective_kernel_data;
  convective_kernel_data.velocity_type      = ConvDiff::TypeVelocityField::DoFVector;
  convective_kernel_data.dof_index_velocity = this->get_dof_index_velocity();
  convective_kernel_data.numerical_flux_formulation =
    ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // diffusive operator:
  // take interior penalty factor of diffusivity of viscous operator, but use polynomial degree of
  // pressure shape functions.
  ConvDiff::Operators::DiffusiveKernelData diffusive_kernel_data;
  diffusive_kernel_data.IP_factor = this->param.IP_factor_viscous;
  // Note: the diffusive operator is initialized with constant viscosity. In case of spatially (and
  // temporally) varying viscosities the diffusive operator has to be extended so that it can deal
  // with variable coefficients (and should be updated in case of time dependent problems before
  // applying the preconditioner).
  diffusive_kernel_data.diffusivity    = this->param.viscosity;
  diffusive_kernel_data.degree         = this->param.get_degree_p();
  diffusive_kernel_data.degree_mapping = this->mapping_degree;

  // combined convection-diffusion operator
  ConvDiff::OperatorData<dim> operator_data;
  operator_data.dof_index  = this->get_dof_index_pressure();
  operator_data.quad_index = this->get_quad_index_pressure();

  operator_data.bc                   = boundary_descriptor;
  operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;

  operator_data.unsteady_problem   = this->unsteady_problem_has_to_be_solved();
  operator_data.convective_problem = this->param.nonlinear_problem_has_to_be_solved();
  operator_data.diffusive_problem  = this->param.viscous_problem();

  operator_data.convective_kernel_data = convective_kernel_data;
  operator_data.diffusive_kernel_data  = diffusive_kernel_data;

  pressure_conv_diff_operator.reset(new ConvDiff::Operator<dim, Number>());
  pressure_conv_diff_operator->reinit(this->get_matrix_free(), this->constraint_p, operator_data);
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
DGNavierStokesCoupled<dim, Number>::update_block_preconditioner()
{
  // momentum block
  preconditioner_momentum->update();

  // pressure block
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
    dst *= 1. / this->momentum_operator.get_scaling_factor_mass_matrix();
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
        check_multigrid(this->momentum_operator,preconditioner);

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
    dst *= this->momentum_operator.get_scaling_factor_mass_matrix();
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

    // I. 1/dt (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= this->momentum_operator.get_scaling_factor_mass_matrix();

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
      this->momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

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
      this->momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

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

    // II. pressure convection-diffusion operator A_p
    if(this->unsteady_problem_has_to_be_solved())
    {
      pressure_conv_diff_operator->set_scaling_factor_mass_matrix(
        this->momentum_operator.get_scaling_factor_mass_matrix());
    }

    if(this->param.nonlinear_problem_has_to_be_solved())
      pressure_conv_diff_operator->set_velocity_ptr(this->convective_kernel->get_velocity());

    pressure_conv_diff_operator->apply(dst, tmp_scp_pressure);

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
        singular = laplace_operator_classical->operator_is_singular();
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
