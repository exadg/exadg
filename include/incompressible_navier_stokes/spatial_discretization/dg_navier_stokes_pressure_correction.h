/*
 * dg_navier_stokes_pressure_correction.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PRESSURE_CORRECTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PRESSURE_CORRECTION_H_

#include "dg_navier_stokes_projection_methods.h"

#include "interface.h"
#include "momentum_operator.h"

#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../preconditioners/multigrid_preconditioner.h"

namespace IncNS
{
template<int dim, typename Number = double>
class DGNavierStokesPressureCorrection : public DGNavierStokesProjectionMethods<dim, Number>,
                                         public Interface::OperatorPressureCorrection<Number>
{
private:
  typedef DGNavierStokesBase<dim, Number> BASE;

  typedef DGNavierStokesProjectionMethods<dim, Number> PROJECTION_METHODS_BASE;

  typedef typename PROJECTION_METHODS_BASE::VectorType VectorType;

  typedef typename PROJECTION_METHODS_BASE::Postprocessor Postprocessor;

  typedef DGNavierStokesPressureCorrection<dim, Number> THIS;

  typedef typename PROJECTION_METHODS_BASE::MultigridNumber MultigridNumber;

public:
  /*
   * Constructor.
   */
  DGNavierStokesPressureCorrection(parallel::Triangulation<dim> const & triangulation,
                                   InputParameters const &              parameters_in,
                                   std::shared_ptr<Postprocessor>       postprocessor_in);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesPressureCorrection();

  void
  setup_solvers(double const &     scaling_factor_time_derivative_term = 1.0,
                VectorType const * velocity                            = nullptr);

  /*
   * Momentum step:
   */

  /*
   * Stokes equations or convective term treated explicitly: solve linear system of equations
   */
  void
  solve_linear_momentum_equation(VectorType &       solution,
                                 VectorType const & rhs,
                                 bool const &       update_preconditioner,
                                 double const &     scaling_factor_mass_matrix_term,
                                 unsigned int &     linear_iterations);

  /*
   * Calculation of right-hand side vector:
   */

  // body force term
  void
  evaluate_add_body_force_term(VectorType & dst, double const evaluation_time) const;

  // viscous term
  void
  rhs_add_viscous_term(VectorType & dst, double const evaluation_time) const;

  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */

  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "initialize_vector_for_newton_solver".
   */
  void
  initialize_vector_for_newton_solver(VectorType & src) const;

  /*
   * This function solves the non-linear system of equations in case of an implicit formulation of
   * the convective term.
   */
  void
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     eval_time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass_matrix_term,
                                    unsigned int &     newton_iterations,
                                    unsigned int &     linear_iterations);

  // apply momentum operator
  void
  apply_momentum_operator(VectorType & dst, VectorType const & src);


  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "evaluate_nonlinear_residual".
   */
  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src);

  void
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     evaluation_time) const;

  /*
   * Projection step.
   */

  // rhs pressure gradient
  void
  rhs_pressure_gradient_term(VectorType & dst, double const evaluation_time) const;

  /*
   * Pressure update step.
   */

  // apply inverse pressure mass matrix
  void
  apply_inverse_pressure_mass_matrix(VectorType & dst, VectorType const & src) const;

  /*
   * pressure Poisson equation.
   */
  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const;

  void
  rhs_ppe_laplace_add(VectorType & dst, double const & evaluation_time) const;

  /*
   * Postprocessing.
   */
  void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    unsigned int const time_step_number) const;

  void
  do_postprocessing_steady_problem(VectorType const & velocity, VectorType const & pressure) const;

private:
  /*
   * Setup of momentum solver (operator, preconditioner, solver).
   */
  void
  setup_momentum_solver(double const &     scaling_factor_time_derivative_term,
                        VectorType const * velocity);

  void
  initialize_momentum_operator(double const &     scaling_factor_time_derivative_term,
                               VectorType const * velocity);

  void
  initialize_momentum_preconditioner();

  void
  initialize_momentum_solver();

  /*
   * Setup of inverse mass matrix operator for pressure.
   */
  void
  setup_inverse_mass_matrix_operator_pressure();

  InverseMassMatrixOperator<dim, 1, Number> inverse_mass_pressure;

  /*
   * Momentum equation.
   */
#ifdef USE_MERGED_MOMENTUM_OPERATOR
  MomentumOperatorMerged<dim, Number> momentum_operator_merged;

  std::shared_ptr<NewtonSolver<VectorType,
                               THIS,
                               MomentumOperatorMerged<dim, Number>,
                               IterativeSolverBase<VectorType>>>
    momentum_newton_solver;
#else
  MomentumOperator<dim, Number> momentum_operator;

  std::shared_ptr<
    NewtonSolver<VectorType, THIS, MomentumOperator<dim, Number>, IterativeSolverBase<VectorType>>>
    momentum_newton_solver;
#endif

  std::shared_ptr<PreconditionerBase<Number>>      momentum_preconditioner;
  std::shared_ptr<IterativeSolverBase<VectorType>> momentum_linear_solver;

  VectorType         temp_vector;
  VectorType const * rhs_vector;

  double evaluation_time;
  double scaling_factor_time_derivative_term;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PRESSURE_CORRECTION_H_ \
        */
