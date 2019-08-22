/*
 * dg_pressure_correction.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_PRESSURE_CORRECTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_PRESSURE_CORRECTION_H_

#include "dg_projection_methods.h"

#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../preconditioners/multigrid_preconditioner_momentum.h"

namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class DGNavierStokesPressureCorrection;

template<int dim, typename Number>
class NonlinearMomentumOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DGNavierStokesPressureCorrection<dim, Number> PDEOperator;

public:
  NonlinearMomentumOperator()
    : pde_operator(nullptr), rhs_vector(nullptr), time(0.0), scaling_factor_mass_matrix(1.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & rhs_vector, double const & time, double const & scaling_factor)
  {
    this->rhs_vector                 = &rhs_vector;
    this->time                       = time;
    this->scaling_factor_mass_matrix = scaling_factor;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'initialize_vector_for_newton_solver'.
   */
  void
  initialize_vector_for_newton_solver(VectorType & src) const
  {
    pde_operator->initialize_vector_velocity(src);
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_nonlinear_residual".
   */
  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src)
  {
    pde_operator->evaluate_nonlinear_residual(
      dst, src, rhs_vector, time, scaling_factor_mass_matrix);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * rhs_vector;
  double             time;
  double             scaling_factor_mass_matrix;
};

template<int dim, typename Number = double>
class DGNavierStokesPressureCorrection : public DGNavierStokesProjectionMethods<dim, Number>,
                                         public Interface::OperatorPressureCorrection<Number>
{
private:
  typedef DGNavierStokesProjectionMethods<dim, Number>  Base;
  typedef DGNavierStokesPressureCorrection<dim, Number> This;

  typedef typename Base::VectorType      VectorType;
  typedef typename Base::Postprocessor   Postprocessor;
  typedef typename Base::MultigridNumber MultigridNumber;

public:
  /*
   * Constructor.
   */
  DGNavierStokesPressureCorrection(parallel::TriangulationBase<dim> const & triangulation,
                                   InputParameters const &              parameters,
                                   std::shared_ptr<Postprocessor>       postprocessor);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesPressureCorrection();

  void
  setup_solvers(double const & scaling_factor_time_derivative_term, VectorType const & velocity);

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

  // viscous term
  void
  rhs_add_viscous_term(VectorType & dst, double const time) const;

  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */
  void
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass_matrix_term,
                                    unsigned int &     newton_iterations,
                                    unsigned int &     linear_iterations);

  /*
   * This function evaluates the nonlinear residual.
   */
  void
  evaluate_nonlinear_residual(VectorType &       dst,
                              VectorType const & src,
                              VectorType const * rhs_vector,
                              double const &     time,
                              double const &     scaling_factor_mass_matrix) const;

  /*
   * This function evaluates the nonlinear residual of the steady Navier-Stokes equations (momentum
   * equation and continuity equation).
   */
  void
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     time) const;

  /*
   * This function applies the linearized momentum operator and is used for throughput measurements.
   */
  void
  apply_momentum_operator(VectorType & dst, VectorType const & src);

  /*
   * Projection step.
   */

  // rhs pressure gradient
  void
  rhs_pressure_gradient_term(VectorType & dst, double const time) const;

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
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

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
  setup_momentum_solver();

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

  // Nonlinear operator and solver
  NonlinearMomentumOperator<dim, Number> nonlinear_operator;

  std::shared_ptr<NewtonSolver<VectorType,
                               NonlinearMomentumOperator<dim, Number>,
                               MomentumOperator<dim, Number>,
                               IterativeSolverBase<VectorType>>>
    momentum_newton_solver;

  // linear solver (momentum_operator serves as linear operator)
  std::shared_ptr<PreconditionerBase<Number>>      momentum_preconditioner;
  std::shared_ptr<IterativeSolverBase<VectorType>> momentum_linear_solver;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_PRESSURE_CORRECTION_H_ \
        */
