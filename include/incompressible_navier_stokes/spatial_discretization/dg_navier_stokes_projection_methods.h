/*
 * dg_navier_stokes_projection_methods.h
 *
 *  Created on: Nov 7, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_

#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"

namespace IncNS
{
/*
 * Base class for projection-type incompressible Navier-Stokes solvers such as the high-order dual
 * splitting (velocity-correction) scheme or pressure correction schemes.
 */
template<int dim, int degree_u, int degree_p, typename Number>
class DGNavierStokesProjectionMethods : public DGNavierStokesBase<dim, degree_u, degree_p, Number>
{
protected:
  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> BASE;

  typedef typename BASE::VectorType VectorType;

  typedef typename BASE::Postprocessor Postprocessor;

public:
  /*
   * Constructor.
   */
  DGNavierStokesProjectionMethods(parallel::distributed::Triangulation<dim> const & triangulation,
                                  InputParameters<dim> const &                      parameters_in,
                                  std::shared_ptr<Postprocessor> postprocessor_in);

  /*
   * Desctructor.
   */
  virtual ~DGNavierStokesProjectionMethods();

  /*
   * This function evaluates the rhs-contribution of the viscous term and adds the result to the
   * dst-vector.
   */
  void
  do_rhs_add_viscous_term(VectorType & dst, double const evaluation_time) const;

  /*
   * Pressure Poisson equation: This function evaluates the inhomogeneous parts of boundary face
   * integrals of the negative Laplace operator and adds the result to the dst-vector.
   */
  void
  do_rhs_ppe_laplace_add(VectorType & dst, double const & evaluation_time) const;

  /*
   * This funtion solves the pressure Poisson equation and returns the number of iterations.
   */
  unsigned int
  do_solve_pressure(VectorType & dst, VectorType const & src) const;

  /*
   * This function evaluates the projection operator (homogeneous part = apply).
   */
  void
  apply_projection_operator(VectorType & dst, VectorType const & src) const;

  /*
   * This function evaluates the Laplace operator (homogeneous part = apply).
   */
  void
  apply_laplace_operator(VectorType & dst, VectorType const & src) const;

protected:
  /*
   * Initializes the preconditioner and solver for the pressure Poisson equation. Can be done in
   * this base class since it is the same for dual-splitting and pressure-correction. The function
   * is declared virtual so that individual initializations required for derived class can be added
   * where needed.
   */
  virtual void
  setup_pressure_poisson_solver();

  // Pressure Poisson equation (operator, preconditioner, solver).
  Poisson::LaplaceOperator<dim, degree_p, Number> laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_poisson;

  std::shared_ptr<IterativeSolverBase<VectorType>> pressure_poisson_solver;

private:
  /*
   * Initialization functions called during setup of pressure Poisson solver.
   */
  void
  initialize_laplace_operator();

  void
  initialize_preconditioner_pressure_poisson();

  void
  initialize_solver_pressure_poisson();
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_ \
        */
