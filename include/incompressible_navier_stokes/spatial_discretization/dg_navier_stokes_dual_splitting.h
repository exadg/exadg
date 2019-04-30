/*
 * dg_navier_stokes_dual_splitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_

// base class
#include "dg_navier_stokes_projection_methods.h"

// boundary conditions specifically related to dual splitting scheme
#include "boundary_conditions_dual_splitting/pressure_neumann_bc_convective_term.h"
#include "boundary_conditions_dual_splitting/pressure_neumann_bc_viscous_term.h"
#include "boundary_conditions_dual_splitting/velocity_divergence_convective_term.h"

#include "../interface_space_time/operator.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p = degree_u - 1, typename Number = double>
class DGNavierStokesDualSplitting
  : public DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>,
    public Interface::OperatorDualSplitting<Number>
{
private:
  typedef DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number> PROJECTION_METHODS_BASE;

  typedef typename PROJECTION_METHODS_BASE::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef typename PROJECTION_METHODS_BASE::Postprocessor Postprocessor;

  typedef DGNavierStokesDualSplitting<dim, degree_u, degree_p, Number> THIS;

public:
  /*
   * Constructor.
   */
  DGNavierStokesDualSplitting(parallel::Triangulation<dim> const & triangulation,
                              InputParameters<dim> const &         parameters_in,
                              std::shared_ptr<Postprocessor>       postprocessor_in);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesDualSplitting();

  void
  setup_solvers(double const & scaling_factor_time_derivative_term);

  /*
   *  Implicit formulation of convective term.
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
  solve_nonlinear_convective_problem(VectorType &       dst,
                                     VectorType const & sum_alphai_ui,
                                     double const &     eval_time,
                                     double const &     scaling_factor_mass_matrix_term,
                                     unsigned int &     newton_iterations,
                                     unsigned int &     linear_iterations);

  /*
   *  The implementation of the Newton solver requires that the underlying operator
   *  implements a function called "evaluate_nonlinear_residual".
   */
  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src);

  /*
   * The implementation of the Newton solver requires that the linearized operator
   * implements a function called "set_solution_linearization".
   */
  void
  set_solution_linearization(VectorType const & solution_linearization);

  /*
   *  To solve the linearized convective problem, the underlying operator has to implement a
   * function called "vmult" (which calculates the matrix vector product for the linearized
   * convective problem).
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  /*
   * This function evaluates the convective term and applies the inverse mass matrix.
   */
  void
  evaluate_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                         VectorType const & src,
                                                         double const       evaluation_time) const;

  /*
   * This function evaluates the body force term and applies the inverse mass matrix.
   */
  void
  evaluate_body_force_and_apply_inverse_mass_matrix(VectorType & dst,
                                                    double const evaluation_time) const;

  /*
   * Pressure Poisson equation.
   */

  // rhs pressure: velocity divergence
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  void
  rhs_velocity_divergence_term(VectorType & dst, double const & evaluation_time) const;

  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & eval_time);

  void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure
  void
  rhs_ppe_nbc_add(VectorType & dst, double const & evaluation_time);

  // rhs pressure: Neumann BC convective term
  void
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure: Neumann BC viscous term
  void
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const;

  void
  rhs_ppe_laplace_add(VectorType & dst, double const & evaluation_time) const;

  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const;


  /*
   * Viscous step.
   */

  void
  apply_helmholtz_operator(VectorType & dst, VectorType const & src) const;

  void
  rhs_add_viscous_term(VectorType & dst, double const evaluation_time) const;

  unsigned int
  solve_viscous(VectorType &       dst,
                VectorType const & src,
                bool const &       update_preconditioner,
                double const &     scaling_factor_time_derivative_term);

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
   * Setup of solvers for the different sub-steps of the dual splitting scheme.
   */
  void
  setup_convective_solver();

  /*
   * Setup of pressure Poisson solver: call function of base class and do additional for dual
   * splitting scheme
   */
  void
  setup_pressure_poisson_solver();

  /*
   * Setup of helmholtz solver (operator, preconditioner, solver).
   */
  void
  setup_helmholtz_solver(double const & scaling_factor_time_derivative_term);

  void
  initialize_helmholtz_operator(double const & scaling_factor_time_derivative_term);

  void
  initialize_helmholtz_preconditioner();

  void
  initialize_helmholtz_solver();

  /*
   * rhs pressure Poisson equation
   */

  // boundary conditions for intermediate velocity u_hat: body force term
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

  // Neumann boundary condition term
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


  /*
   * Viscous step (Helmholtz-like equation).
   */
  MomentumOperator<dim, degree_u, Number> helmholtz_operator;

  std::shared_ptr<PreconditionerBase<Number>> helmholtz_preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> helmholtz_solver;

  /*
   * Implicit solution of convective step (solve non-linear system of equations).
   */
  VectorType         temp;
  VectorType const * sum_alphai_ui;

  // implicit solution of convective step
  std::shared_ptr<InverseMassMatrixPreconditioner<dim, degree_u, Number, dim>>
    preconditioner_convective_problem;

  std::shared_ptr<IterativeSolverBase<VectorType>> linear_solver;
  std::shared_ptr<NewtonSolver<VectorType, THIS, THIS, IterativeSolverBase<VectorType>>>
    newton_solver;

  /*
   * pressure Poisson equation: Calculation of right-hand side vector
   */

  // velocity divergence term
  VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>
    velocity_divergence_convective_term;

  // pressure Neumann boundary condition
  PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number> pressure_nbc_convective_term;
  PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>    pressure_nbc_viscous_term;

  /*
   * Element variable used to store the current pyhsical time. Note that this variable is not only
   * needed in case of an implicit treatment of the convective term, but also for the evaluation of
   * the right-hand side of the pressure Poisson equation.
   */
  double evaluation_time;

  /*
   * This factor is needed as element variable in case of an implicit treatment of the convective
   * term where a nonlinear system of equations has to be solved.
   */
  double scaling_factor_time_derivative_term;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_DUAL_SPLITTING_H_ \
        */
