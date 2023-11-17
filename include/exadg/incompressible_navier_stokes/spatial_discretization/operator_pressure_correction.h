/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver.h>

namespace ExaDG
{
namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class OperatorPressureCorrection;

template<int dim, typename Number>
class NonlinearMomentumOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef OperatorPressureCorrection<dim, Number> PDEOperator;

public:
  NonlinearMomentumOperator()
    : pde_operator(nullptr), rhs_vector(nullptr), time(0.0), scaling_factor_mass(1.0)
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
    this->rhs_vector          = &rhs_vector;
    this->time                = time;
    this->scaling_factor_mass = scaling_factor;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_residual'.
   */
  void
  evaluate_residual(VectorType & dst, VectorType const & src)
  {
    pde_operator->evaluate_nonlinear_residual(dst, src, rhs_vector, time, scaling_factor_mass);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * rhs_vector;
  double             time;
  double             scaling_factor_mass;
};

template<int dim, typename Number = double>
class OperatorPressureCorrection : public OperatorProjectionMethods<dim, Number>
{
private:
  typedef SpatialOperatorBase<dim, Number>        Base;
  typedef OperatorProjectionMethods<dim, Number>  ProjectionBase;
  typedef OperatorPressureCorrection<dim, Number> This;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::scalar scalar;
  typedef typename Base::vector vector;
  typedef typename Base::tensor tensor;

  typedef typename Base::Range Range;

  typedef typename Base::FaceIntegratorU FaceIntegratorU;
  typedef typename Base::FaceIntegratorP FaceIntegratorP;

public:
  /*
   * Constructor.
   */
  OperatorPressureCorrection(std::shared_ptr<Grid<dim> const>               grid,
                             std::shared_ptr<dealii::Mapping<dim> const>    mapping,
                             std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                             std::shared_ptr<FieldFunctions<dim> const>     field_functions,
                             Parameters const &                             parameters,
                             std::string const &                            field,
                             MPI_Comm const &                               mpi_comm);

  /*
   * Destructor.
   */
  virtual ~OperatorPressureCorrection();

  /*
   * Initializes the inverse pressure mass matrix operator needed for the pressure correction
   * scheme, as well as the pressure mass operator needed in the ALE case only (where the mass
   * operator may be evaluated at different times depending on the specific ALE formulation chosen).
   */
  void
  setup_derived() final;

  void
  setup_solvers(double const & scaling_factor_mass, VectorType const & velocity) final;

  void
  update_after_grid_motion(bool const update_matrix_free) final;

  /*
   * Momentum step:
   */

  /*
   * Stokes equations or convective term treated explicitly: solve linear system of equations
   */
  unsigned int
  solve_linear_momentum_equation(VectorType &       solution,
                                 VectorType const & rhs,
                                 bool const &       update_preconditioner,
                                 double const &     scaling_factor_mass);

  /*
   * Calculation of right-hand side vector:
   */

  // viscous term
  void
  rhs_add_viscous_term(VectorType & dst, double const time) const;

  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass);

  /*
   * This function evaluates the nonlinear residual.
   */
  void
  evaluate_nonlinear_residual(VectorType &       dst,
                              VectorType const & src,
                              VectorType const * rhs_vector,
                              double const &     time,
                              double const &     scaling_factor_mass) const;

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
  rhs_pressure_gradient_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                          VectorType const & pressure) const;

  void
  evaluate_pressure_gradient_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                               VectorType const & src,
                                                               VectorType const & pressure) const;

  /*
   * Pressure update step.
   */

  // apply inverse pressure mass operator
  void
  apply_inverse_pressure_mass_operator(VectorType & dst, VectorType const & src) const;

  /*
   * pressure Poisson equation.
   */
  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src, bool const update_preconditioner) const;

  void
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

  void
  rhs_ppe_laplace_add_dirichlet_bc_from_dof_vector(VectorType & dst, VectorType const & src) const;

  void
  interpolate_pressure_dirichlet_bc(VectorType & dst, double const & time) const;

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
   * Setup of inverse mass operator for pressure.
   */
  void
  setup_inverse_mass_operator_pressure();

  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  local_interpolate_pressure_dirichlet_bc_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  InverseMassOperator<dim, 1, Number> inverse_mass_pressure;

  /*
   * Momentum equation.
   */

  // Nonlinear operator and solver
  NonlinearMomentumOperator<dim, Number> nonlinear_operator;

  std::shared_ptr<Newton::Solver<VectorType,
                                 NonlinearMomentumOperator<dim, Number>,
                                 MomentumOperator<dim, Number>,
                                 Krylov::SolverBase<VectorType>>>
    momentum_newton_solver;

  // linear solver (momentum_operator serves as linear operator)
  std::shared_ptr<PreconditionerBase<Number>>     momentum_preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> momentum_linear_solver;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_ \
        */
