/*
 * dg_navier_stokes_dual_splitting.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_dual_splitting.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesDualSplitting<dim, Number>::DGNavierStokesDualSplitting(
  parallel::Triangulation<dim> const & triangulation,
  InputParameters const &              parameters_in,
  std::shared_ptr<Postprocessor>       postprocessor_in)
  : PROJECTION_METHODS_BASE(triangulation, parameters_in, postprocessor_in),
    sum_alphai_ui(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0)
{
}

template<int dim, typename Number>
DGNavierStokesDualSplitting<dim, Number>::~DGNavierStokesDualSplitting()
{
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::setup_solvers(
  double const &     scaling_factor_time_derivative_term,
  VectorType const * velocity)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  // initialize vectors that are needed by the nonlinear solver
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    setup_convective_solver(velocity);
  }

  this->setup_pressure_poisson_solver();

  this->setup_projection_solver();

  setup_helmholtz_solver(scaling_factor_time_derivative_term);

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::setup_convective_solver(VectorType const * velocity)
{
  this->initialize_vector_velocity(temp);

  AssertThrow(
    velocity != nullptr,
    ExcMessage(
      "To initialize preconditioners, convective kernel needs access to a velocity field."));

  this->set_velocity_ptr(*velocity);

  // preconditioner implicit convective step
  preconditioner_convective_problem.reset(
    new InverseMassMatrixPreconditioner<dim, dim, Number>(this->matrix_free,
                                                          this->param.degree_u,
                                                          this->get_dof_index_velocity(),
                                                          this->get_quad_index_velocity_linear()));

  // linear solver (GMRES)
  GMRESSolverData solver_data;
  solver_data.max_iter             = this->param.solver_data_convective.max_iter;
  solver_data.solver_tolerance_abs = this->param.solver_data_convective.abs_tol;
  solver_data.solver_tolerance_rel = this->param.solver_data_convective.rel_tol;
  solver_data.max_n_tmp_vectors    = this->param.solver_data_convective.max_krylov_size;

  // always use inverse mass matrix preconditioner
  solver_data.use_preconditioner = true;

  // setup linear solver
  linear_solver.reset(new GMRESSolver<THIS, PreconditionerBase<Number>, VectorType>(
    *this, *preconditioner_convective_problem, solver_data));

  // setup Newton solver
  newton_solver.reset(new NewtonSolver<VectorType, THIS, THIS, IterativeSolverBase<VectorType>>(
    this->param.newton_solver_data_convective, *this, *this, *linear_solver));
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::setup_helmholtz_solver(
  double const & scaling_factor_time_derivative_term)
{
  initialize_helmholtz_operator(scaling_factor_time_derivative_term);

  initialize_helmholtz_preconditioner();

  initialize_helmholtz_solver();
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_helmholtz_operator(
  double const & scaling_factor_time_derivative_term)
{
#ifdef USE_MERGED_MOMENTUM_OPERATOR
  MomentumOperatorMergedData<dim> data;

  // the dual splitting scheme is an unsteady solver
  data.unsteady_problem           = true;
  data.scaling_factor_mass_matrix = scaling_factor_time_derivative_term;

  // convective problem
  data.convective_problem = false; // dual splitting scheme!

  if(this->param.viscous_problem())
    data.viscous_problem = true;

  data.viscous_kernel_data    = this->viscous_kernel_data;
  data.convective_kernel_data = this->convective_kernel_data;

  data.mg_operator_type = MultigridOperatorType::ReactionDiffusion; // dual splitting scheme!

  data.bc = this->boundary_descriptor_velocity;

  data.dof_index  = this->get_dof_index_velocity();
  data.quad_index = this->get_quad_index_velocity_linearized();

  data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;

  AffineConstraints<double> constraint_dummy;
  constraint_dummy.close();

  momentum_operator.reinit(
    this->matrix_free, constraint_dummy, data, this->viscous_kernel, this->convective_kernel);
#else
  // setup velocity convection-diffusion operator
  MomentumOperatorData<dim> momentum_operator_data;

  // unsteady problem
  momentum_operator_data.unsteady_problem = true;

  momentum_operator_data.scaling_factor_time_derivative_term = scaling_factor_time_derivative_term;

  // convective problem = false (dual splitting scheme!)
  momentum_operator_data.convective_problem = false;

  momentum_operator_data.dof_index       = this->get_dof_index_velocity();
  momentum_operator_data.quad_index_std  = this->get_quad_index_velocity_linear();
  momentum_operator_data.quad_index_over = this->get_quad_index_velocity_nonlinear();

  momentum_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  momentum_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;
  // dual splitting scheme: We don't have a choice here!
  momentum_operator_data.mg_operator_type = MultigridOperatorType::ReactionDiffusion;

  momentum_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  momentum_operator_data.viscous_operator_data     = this->viscous_operator_data;
  momentum_operator_data.convective_operator_data  = this->convective_operator_data;

  helmholtz_operator.reinit(this->get_matrix_free(),
                            momentum_operator_data,
                            this->mass_matrix_operator,
                            this->viscous_operator,
                            this->convective_operator);
#endif
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_helmholtz_preconditioner()
{
  if(this->param.preconditioner_viscous == PreconditionerViscous::None)
  {
    // do nothing, preconditioner will not be used
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    helmholtz_preconditioner.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
      this->matrix_free,
      this->param.degree_u,
      this->get_dof_index_velocity(),
      this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi)
  {
#ifdef USE_MERGED_MOMENTUM_OPERATOR
    helmholtz_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperatorMerged<dim, Number>>(momentum_operator));
#else
    helmholtz_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(helmholtz_operator));
#endif
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi)
  {
#ifdef USE_MERGED_MOMENTUM_OPERATOR
    helmholtz_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperatorMerged<dim, Number>>(momentum_operator));
#else
    helmholtz_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(helmholtz_operator));
#endif
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
  {
#ifdef USE_MERGED_MOMENTUM_OPERATOR
    typedef MultigridPreconditionerMerged<dim, Number, MultigridNumber> MULTIGRID;

    helmholtz_preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(helmholtz_preconditioner);

    auto & dof_handler = this->get_dof_handler_u();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(this->param.multigrid_data_viscous,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  momentum_operator.get_operator_data());
#else
    typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    helmholtz_preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(helmholtz_preconditioner);

    auto & dof_handler = this->get_dof_handler_u();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(this->param.multigrid_data_viscous,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  helmholtz_operator.get_operator_data());
#endif
  }
  else
  {
    AssertThrow(false, ExcMessage("Preconditioner specified for viscous step is not implemented."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_helmholtz_solver()
{
  if(this->param.solver_viscous == SolverViscous::CG)
  {
    // setup solver data
    CGSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;

    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

#ifdef USE_MERGED_MOMENTUM_OPERATOR
    helmholtz_solver.reset(
      new CGSolver<MomentumOperatorMerged<dim, Number>, PreconditionerBase<Number>, VectorType>(
        momentum_operator, *helmholtz_preconditioner, solver_data));
#else
    helmholtz_solver.reset(
      new CGSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        helmholtz_operator, *helmholtz_preconditioner, solver_data));
#endif
  }
  else if(this->param.solver_viscous == SolverViscous::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_viscous.max_krylov_size;
    // use default value of compute_eigenvalues

    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

#ifdef USE_MERGED_MOMENTUM_OPERATOR
    helmholtz_solver.reset(
      new GMRESSolver<MomentumOperatorMerged<dim, Number>, PreconditionerBase<Number>, VectorType>(
        momentum_operator, *helmholtz_preconditioner, solver_data));
#else
    helmholtz_solver.reset(
      new GMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        helmholtz_operator, *helmholtz_preconditioner, solver_data));
#endif
  }
  else if(this->param.solver_viscous == SolverViscous::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_viscous.max_krylov_size;

    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

#ifdef USE_MERGED_MOMENTUM_OPERATOR
    helmholtz_solver.reset(
      new FGMRESSolver<MomentumOperatorMerged<dim, Number>, PreconditionerBase<Number>, VectorType>(
        momentum_operator, *helmholtz_preconditioner, solver_data));
#else
    helmholtz_solver.reset(
      new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        helmholtz_operator, *helmholtz_preconditioner, solver_data));
#endif
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified viscous solver is not implemented."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_vector_for_newton_solver(
  VectorType & src) const
{
  this->initialize_vector_velocity(src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::set_solution_linearization(
  VectorType const & solution_linearization)
{
  this->convective_kernel->set_velocity_ptr(solution_linearization);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::solve_nonlinear_convective_problem(
  VectorType &       dst,
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
  newton_solver->solve(
    dst, newton_iterations, linear_iterations, /* update_preconditioner = */ false, 1);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::evaluate_nonlinear_residual(VectorType &       dst,
                                                                      VectorType const & src)
{
  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst, evaluation_time);
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

  this->convective_operator.evaluate_nonlinear_operator_add(dst, src, evaluation_time);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  this->mass_matrix_operator.apply(dst, src);

  dst *= scaling_factor_time_derivative_term;

  this->convective_operator.set_evaluation_time(evaluation_time);
  this->convective_operator.apply_add(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::evaluate_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  double const       evaluation_time) const
{
  this->convective_operator.evaluate_nonlinear_operator(dst, src, evaluation_time);
  this->inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::evaluate_body_force_and_apply_inverse_mass_matrix(
  VectorType & dst,
  double const evaluation_time) const
{
  this->rhs_operator.evaluate(dst, evaluation_time);

  this->inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::apply_velocity_divergence_term(
  VectorType &       dst,
  VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_velocity_divergence_term(
  VectorType &   dst,
  double const & evaluation_time) const
{
  this->divergence_operator.rhs(dst, evaluation_time);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_div_term_body_forces_add(VectorType &   dst,
                                                                           double const & eval_time)
{
  evaluation_time = eval_time;

  VectorType src_dummy;
  this->matrix_free.loop(&THIS::cell_loop_empty,
                         &THIS::face_loop_empty,
                         &THIS::local_rhs_ppe_div_term_body_forces_boundary_face,
                         this,
                         dst,
                         src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(matrix_free, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor_velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs = evaluate_vectorial_function(this->field_functions->right_hand_side,
                                                 q_points,
                                                 evaluation_time);

        scalar flux_times_normal = rhs * integrator.get_normal_vector(q);
        // minus sign is introduced here which allows to call a function of type ...add()
        // and avoids a scaling of the resulting vector by the factor -1.0
        integrator.submit_value(-flux_times_normal, q);
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
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(true, false);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_div_term_convective_term_add(
  VectorType &       dst,
  VectorType const & src) const
{
  this->matrix_free.loop(&THIS::cell_loop_empty,
                         &THIS::face_loop_empty,
                         &THIS::local_rhs_ppe_div_term_convective_term_boundary_face,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_div_term_convective_term_boundary_face(
  MatrixFree<dim, Number> const &               matrix_free,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_nonlinear();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src, true, true);

    pressure.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor_velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);
        scalar div_u  = velocity.get_divergence(q);

        scalar flux_times_normal = (grad_u * u + div_u * u) * normal;

        pressure.submit_value(flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and Symmetry boundaries.
        // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in
        // case of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
        // prescribe a consistent boundary condition for g_{u_hat} derived from the convective
        // step of the dual splitting scheme which differs from the DBC g_u. Applying this
        // consistent DBC to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry,
        // we obtain g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous
        // contributions for g_{u_hat}*n.
        scalar zero = make_vectorized_array<Number>(0.0);
        pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_nbc_add(VectorType &   dst,
                                                          double const & eval_time)
{
  evaluation_time = eval_time;

  VectorType src_dummy;
  this->matrix_free.loop(&THIS::cell_loop_empty,
                         &THIS::face_loop_empty,
                         &THIS::local_rhs_ppe_nbc_add_boundary_face,
                         this,
                         dst,
                         src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_add_boundary_face(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(data, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP      boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs = evaluate_vectorial_function(this->field_functions->right_hand_side,
                                                 q_points,
                                                 evaluation_time);

        // evaluate boundary condition
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
        vector dudt = evaluate_vectorial_function(it->second, q_points, evaluation_time);

        vector normal = integrator.get_normal_vector(q);

        scalar h = -normal * (dudt - rhs);

        integrator.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(true, false);
    integrator.distribute_local_to_global(dst);
  }
}


template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_convective_add(VectorType &       dst,
                                                                 VectorType const & src) const
{
  this->matrix_free.loop(&THIS::cell_loop_empty,
                         &THIS::face_loop_empty,
                         &THIS::local_rhs_ppe_nbc_convective_add_boundary_face,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_convective_add_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_nonlinear();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src, true, true);

    pressure.reinit(face);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = make_vectorized_array<Number>(0.0);

        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);
        scalar div_u  = velocity.get_divergence(q);

        h = -normal * (grad_u * u + div_u * u);

        pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }

    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_viscous_add(VectorType &       dst,
                                                              VectorType const & src) const
{
  this->matrix_free.loop(&THIS::cell_loop_empty,
                         &THIS::face_loop_empty,
                         &THIS::local_rhs_ppe_nbc_viscous_add_boundary_face,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_viscous_add_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_quad_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_linear();

  FaceIntegratorU omega(matrix_free, true, dof_index_velocity, quad_index);

  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure.reinit(face);

    omega.reinit(face);
    omega.gather_evaluate(src, false, true);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      scalar viscosity = this->get_viscosity_boundary_face(face, q);

      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = make_vectorized_array<Number>(0.0);

        vector normal = pressure.get_normal_vector(q);

        vector curl_omega = CurlCompute<dim, FaceIntegratorU>::compute(omega, q);

        h = -normal * (viscosity * curl_omega);

        pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                              double const & evaluation_time) const
{
  PROJECTION_METHODS_BASE::do_rhs_ppe_laplace_add(dst, evaluation_time);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesDualSplitting<dim, Number>::solve_pressure(VectorType &       dst,
                                                         VectorType const & src) const
{
  return PROJECTION_METHODS_BASE::do_solve_pressure(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_add_viscous_term(VectorType & dst,
                                                               double const evaluation_time) const
{
  PROJECTION_METHODS_BASE::do_rhs_add_viscous_term(dst, evaluation_time);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesDualSplitting<dim, Number>::solve_viscous(VectorType &       dst,
                                                        VectorType const & src,
                                                        bool const &       update_preconditioner,
                                                        double const &     factor)
{
  // Update Helmholtz operator
#ifdef USE_MERGED_MOMENTUM_OPERATOR
  momentum_operator.set_scaling_factor_mass_matrix(factor);
#else
  helmholtz_operator.set_scaling_factor_time_derivative_term(factor);
#endif

  unsigned int n_iter = helmholtz_solver->solve(dst, src, update_preconditioner);

  return n_iter;
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::apply_helmholtz_operator(VectorType &       dst,
                                                                   VectorType const & src) const
{
#ifdef USE_MERGED_MOMENTUM_OPERATOR
  momentum_operator.vmult(dst, src);
#else
  helmholtz_operator.vmult(dst, src);
#endif
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::do_postprocessing(
  VectorType const & velocity,
  VectorType const & pressure,
  double const       time,
  unsigned int const time_step_number) const
{
  bool const standard = true;
  if(standard)
  {
    this->postprocessor->do_postprocessing(velocity, pressure, time, time_step_number);
  }
  else // consider pressure error and velocity error
  {
    VectorType velocity_error;
    this->initialize_vector_velocity(velocity_error);

    VectorType pressure_error;
    this->initialize_vector_pressure(pressure_error);

    this->prescribe_initial_conditions(velocity_error, pressure_error, time);

    velocity_error.add(-1.0, velocity);
    pressure_error.add(-1.0, pressure);

    this->postprocessor->do_postprocessing(velocity_error, pressure_error, time, time_step_number);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::do_postprocessing_steady_problem(
  VectorType const & velocity,
  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity, pressure);
}

template class DGNavierStokesDualSplitting<2, float>;
template class DGNavierStokesDualSplitting<2, double>;

template class DGNavierStokesDualSplitting<3, float>;
template class DGNavierStokesDualSplitting<3, double>;

} // namespace IncNS
