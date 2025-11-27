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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class OperatorProjectionMethods;

template<int dim, typename Number>
class NonlinearMomentumOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef OperatorProjectionMethods<dim, Number> PDEOperator;

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

/*
 * Base class for projection-type incompressible Navier-Stokes solvers such as the high-order dual
 * splitting (velocity-correction) scheme or pressure correction schemes.
 */
template<int dim, typename Number>
class OperatorProjectionMethods : public SpatialOperatorBase<dim, Number>
{
protected:
  typedef SpatialOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType       VectorType;
  typedef typename Base::MultigridPoisson MultigridPoisson;

public:
  /*
   * Constructor.
   */
  OperatorProjectionMethods(
    std::shared_ptr<Grid<dim> const>                      grid,
    std::shared_ptr<dealii::Mapping<dim> const>           mapping,
    std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
    std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor,
    std::shared_ptr<FieldFunctions<dim> const>            field_functions,
    Parameters const &                                    parameters,
    std::string const &                                   field,
    MPI_Comm const &                                      mpi_comm);

  /*
   * Destructor.
   */
  virtual ~OperatorProjectionMethods();

protected:
  /*
   * Calls setup() function of base class and additionally initializes the pressure Poisson operator
   * needed for projection-type methods.
   */
  void
  setup_derived() override;

  void
  setup_preconditioners_and_solvers() override;

public:
  void
  update_after_grid_motion(bool const update_matrix_free) override;

  /*
   * Pressure Poisson equation: This function evaluates the inhomogeneous parts of boundary face
   * integrals of the negative Laplace operator and adds the result to the dst-vector.
   */
  void
  do_rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

  /*
   * This function solves the pressure Poisson equation and returns the number of iterations.
   */
  unsigned int
  do_solve_pressure(VectorType &       dst,
                    VectorType const & src,
                    bool const         update_preconditioner) const;

  /*
   * This function applies the projection operator (used for throughput measurements).
   */
  void
  apply_projection_operator(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the Laplace operator (used for throughput measurements).
   */
  void
  apply_laplace_operator(VectorType & dst, VectorType const & src) const;

  /*
   * Momentum step:
   */

  /*
   * Momentum step is a linear system of equations
   */
  unsigned int
  solve_linear_momentum_equation(VectorType &       solution,
                                 VectorType const & rhs,
                                 VectorType const & transport_velocity,
                                 bool const &       update_preconditioner,
                                 double const &     scaling_factor_mass);

  /*
   * This function evaluates the rhs-contribution of the viscous term and adds the result to the
   * dst-vector.
   */
  void
  rhs_add_viscous_term(VectorType & dst, double const time) const;

  /*
   * This function evaluates the rhs-contribution of the convective term and adds the result to the
   * dst-vector. This functions is only used for a "linearly implicit formulation" of the convective
   * term.
   */
  void
  rhs_add_convective_term(VectorType &       dst,
                          VectorType const & transport_velocity,
                          double const       time) const;

  /*
   * Momentum step is a non-linear system of equations
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass);

  /*
   * This function evaluates the linearized residual.
   */
  void
  evaluate_linearized_residual(VectorType &       dst,
                               VectorType const & src,
                               VectorType const & transport_velocity,
                               VectorType const * rhs_vector,
                               double const &     time,
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

protected:
  // Pressure Poisson equation (operator, preconditioner, solver).
  Poisson::LaplaceOperator<dim, Number, 1> laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_poisson;

  std::shared_ptr<Krylov::SolverBase<VectorType>> pressure_poisson_solver;

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

  void
  setup_momentum_preconditioner();

  void
  setup_momentum_solver();

private:
  void
  initialize_laplace_operator();

  /*
   * Setup functions called during setup of pressure Poisson solver.
   */
  void
  setup_preconditioner_pressure_poisson();

  void
  setup_solver_pressure_poisson();
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_ \
        */
