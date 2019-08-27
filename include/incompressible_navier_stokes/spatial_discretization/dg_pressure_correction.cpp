/*
 * dg_pressure_correction.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_pressure_correction.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesPressureCorrection<dim, Number>::DGNavierStokesPressureCorrection(
  parallel::TriangulationBase<dim> const & triangulation,
  InputParameters const &              parameters,
  std::shared_ptr<Postprocessor>       postprocessor)
  : Base(triangulation, parameters, postprocessor)
{
}

template<int dim, typename Number>
DGNavierStokesPressureCorrection<dim, Number>::~DGNavierStokesPressureCorrection()
{
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::setup_solvers(
  double const &     scaling_factor_time_derivative_term,
  VectorType const & velocity)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  Base::setup_solvers(scaling_factor_time_derivative_term, velocity);

  setup_momentum_solver();

  Base::setup_pressure_poisson_solver();

  Base::setup_projection_solver();

  setup_inverse_mass_matrix_operator_pressure();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::setup_momentum_solver()
{
  initialize_momentum_preconditioner();

  initialize_momentum_solver();
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::initialize_momentum_preconditioner()
{
  if(this->param.preconditioner_momentum == MomentumPreconditioner::InverseMassMatrix)
  {
    momentum_preconditioner.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
      this->matrix_free,
      this->param.degree_u,
      this->get_dof_index_velocity(),
      this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::PointJacobi)
  {
    momentum_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::BlockJacobi)
  {
    momentum_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::Multigrid)
  {
    typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    momentum_preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(momentum_preconditioner);


    auto & dof_handler = this->get_dof_handler_u();

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  this->momentum_operator,
                                  this->param.multigrid_operator_type_momentum);
  }
  else
  {
    AssertThrow(this->param.preconditioner_momentum == MomentumPreconditioner::None,
                ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::initialize_momentum_solver()
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
      new CGSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *momentum_preconditioner, solver_data));
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
      new GMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *momentum_preconditioner, solver_data));
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
      new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *momentum_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for momentum equation is not implemented."));
  }


  // Navier-Stokes equations with an implicit treatment of the convective term
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // nonlinear_operator;
    nonlinear_operator.initialize(*this);

    // setup Newton solver
    momentum_newton_solver.reset(
      new NewtonSolver<VectorType,
                       NonlinearMomentumOperator<dim, Number>,
                       MomentumOperator<dim, Number>,
                       IterativeSolverBase<VectorType>>(this->param.newton_solver_data_momentum,
                                                        nonlinear_operator,
                                                        this->momentum_operator,
                                                        *momentum_linear_solver));
  }
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::setup_inverse_mass_matrix_operator_pressure()
{
  // inverse mass matrix operator pressure (needed for pressure update in case of rotational
  // formulation)
  inverse_mass_pressure.initialize(this->matrix_free,
                                   this->param.get_degree_p(),
                                   this->get_dof_index_pressure(),
                                   this->get_quad_index_pressure());
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::solve_linear_momentum_equation(
  VectorType &       solution,
  VectorType const & rhs,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass_matrix_term,
  unsigned int &     linear_iterations)
{
  this->momentum_operator.set_scaling_factor_mass_matrix(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the momentum_operator
  // in this because because this function is only called if the convective term is not considered
  // in the momentum_operator (Stokes eq. or explicit treatment of convective term).

  linear_iterations = momentum_linear_solver->solve(solution, rhs, update_preconditioner);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::rhs_add_viscous_term(VectorType & dst,
                                                                    double const time) const
{
  Base::do_rhs_add_viscous_term(dst, time);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::solve_nonlinear_momentum_equation(
  VectorType &       dst,
  VectorType const & rhs_vector,
  double const &     time,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass_matrix_term,
  unsigned int &     newton_iterations,
  unsigned int &     linear_iterations)
{
  // update nonlinear operator
  nonlinear_operator.update(rhs_vector, time, scaling_factor_mass_matrix_term);

  // Set time and mass matrix scaling factor for linear operator
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_matrix(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  momentum_newton_solver->solve(dst,
                                newton_iterations,
                                linear_iterations,
                                update_preconditioner,
                                this->param.update_preconditioner_momentum_every_newton_iter);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::evaluate_nonlinear_residual(
  VectorType &       dst,
  VectorType const & src,
  VectorType const * rhs_vector,
  double const &     time,
  double const &     scaling_factor_mass_matrix) const
{
  this->mass_matrix_operator.apply_scale(dst, scaling_factor_mass_matrix, src);

  // always evaluate convective term since this function is only called
  // if a nonlinear problem has to be solved, i.e., if the convective operator
  // has to be considered
  this->convective_operator.evaluate_nonlinear_operator_add(dst, src, time);

  // viscous term
  this->viscous_operator.set_time(time);
  this->viscous_operator.evaluate_add(dst, src);

  // rhs vector
  dst.add(-1.0, *rhs_vector);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::evaluate_nonlinear_residual_steady(
  VectorType &       dst_u,
  VectorType &       dst_p,
  VectorType const & src_u,
  VectorType const & src_p,
  double const &     time) const
{
  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst_u, time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.convective_problem())
    this->convective_operator.evaluate_nonlinear_operator_add(dst_u, src_u, time);

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst_u, src_u);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::apply_momentum_operator(VectorType &       dst,
                                                                       VectorType const & src)
{
  this->momentum_operator.apply(dst, src);
}


template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::rhs_pressure_gradient_term(VectorType & dst,
                                                                          double const time) const
{
  this->gradient_operator.rhs(dst, time);
}


template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::apply_inverse_pressure_mass_matrix(
  VectorType &       dst,
  VectorType const & src) const
{
  inverse_mass_pressure.apply(dst, src);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesPressureCorrection<dim, Number>::solve_pressure(VectorType &       dst,
                                                              VectorType const & src) const
{
  return Base::do_solve_pressure(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                                   double const & time) const
{
  Base::do_rhs_ppe_laplace_add(dst, time);
}

template<int dim, typename Number>
void
DGNavierStokesPressureCorrection<dim, Number>::do_postprocessing(
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
DGNavierStokesPressureCorrection<dim, Number>::do_postprocessing_steady_problem(
  VectorType const & velocity,
  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity, pressure);
}

template class DGNavierStokesPressureCorrection<2, float>;
template class DGNavierStokesPressureCorrection<2, double>;

template class DGNavierStokesPressureCorrection<3, float>;
template class DGNavierStokesPressureCorrection<3, double>;

} // namespace IncNS
