/*
 * dg_dual_splitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_DUAL_SPLITTING_H_

// base class
#include "dg_projection_methods.h"

#include "curl_compute.h"

namespace IncNS
{
template<int dim, typename Number = double>
class DGNavierStokesDualSplitting : public DGNavierStokesProjectionMethods<dim, Number>,
                                    public Interface::OperatorDualSplitting<Number>
{
private:
  typedef DGNavierStokesProjectionMethods<dim, Number> Base;
  typedef DGNavierStokesDualSplitting<dim, Number>     This;

  typedef typename Base::VectorType      VectorType;
  typedef typename Base::Postprocessor   Postprocessor;
  typedef typename Base::MultigridNumber MultigridNumber;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorP;

public:
  /*
   * Constructor.
   */
  DGNavierStokesDualSplitting(parallel::TriangulationBase<dim> const & triangulation,
                              InputParameters const &              parameters,
                              std::shared_ptr<Postprocessor>       postprocessor)
    : Base(triangulation, parameters, postprocessor), time(0.0)
  {
  }

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesDualSplitting()
  {
  }

  void
  setup_solvers(double const & scaling_factor_time_derivative_term, VectorType const & velocity);

  /*
   * This function evaluates the convective term and applies the inverse mass matrix.
   */
  void
  evaluate_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                         VectorType const & src,
                                                         double const       time) const;

  /*
   * This function evaluates the body force term and applies the inverse mass matrix.
   */
  void
  evaluate_body_force_and_apply_inverse_mass_matrix(VectorType & dst, double const time) const;

  /*
   * Pressure Poisson equation.
   */

  // rhs pressure: velocity divergence
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  void
  rhs_velocity_divergence_term(VectorType & dst, double const & time) const;

  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & time);

  void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure
  void
  rhs_ppe_nbc_add(VectorType & dst, double const & time);

  // rhs pressure: Neumann BC convective term
  void
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure: Neumann BC viscous term
  void
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const;

  void
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const;


  /*
   * Viscous step.
   */

  void
  apply_helmholtz_operator(VectorType & dst, VectorType const & src) const;

  void
  rhs_add_viscous_term(VectorType & dst, double const time) const;

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
   * Setup of Helmholtz solver (operator, preconditioner, solver).
   */
  void
  setup_helmholtz_solver();

  void
  initialize_helmholtz_preconditioner();

  void
  initialize_helmholtz_solver();

  /*
   * rhs pressure Poisson equation
   */

  void
  cell_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  // rhs PPE: velocity divergence term

  // convective term
  void
  local_rhs_ppe_div_term_convective_term_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // body force term
  void
  local_rhs_ppe_div_term_body_forces_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // Neumann boundary condition term

  // du/dt term and body force term
  void
  local_rhs_ppe_nbc_add_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // convective term
  void
  local_rhs_ppe_nbc_convective_add_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // viscous term
  void
  local_rhs_ppe_nbc_viscous_add_boundary_face(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & face_range) const;


  /*
   * Viscous step (Helmholtz-like equation).
   */
  std::shared_ptr<PreconditionerBase<Number>> helmholtz_preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> helmholtz_solver;

  /*
   * Element variable used to store the current physical time. This variable is needed for the
   * evaluation of the right-hand side of the pressure Poisson equation.
   */
  double time;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_DUAL_SPLITTING_H_ \
        */
