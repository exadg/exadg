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
class DGNavierStokesDualSplitting : public DGNavierStokesProjectionMethods<dim, Number>
{
private:
  typedef DGNavierStokesBase<dim, Number>              Base;
  typedef DGNavierStokesProjectionMethods<dim, Number> ProjBase;
  typedef DGNavierStokesDualSplitting<dim, Number>     This;

  typedef typename Base::VectorType      VectorType;
  typedef typename Base::MultigridNumber MultigridNumber;

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
  DGNavierStokesDualSplitting(
    parallel::TriangulationBase<dim> const & triangulation_in,
    Mapping<dim> const &                     mapping_in,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                    periodic_face_pairs_in,
    std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
    std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
    std::shared_ptr<FieldFunctions<dim>> const      field_functions_in,
    InputParameters const &                         parameters_in,
    MPI_Comm const &                                mpi_comm_in);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesDualSplitting();

  void
  setup_solvers(double const & scaling_factor_time_derivative_term, VectorType const & velocity);

  /*
   * Pressure Poisson equation.
   */

  // rhs pressure: velocity divergence
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  void
  rhs_velocity_divergence_term(VectorType & dst, double const & time) const;

  void
  rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                            VectorType const & velocity) const;

  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & time);

  void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure
  void
  rhs_ppe_nbc_body_force_term_add(VectorType & dst, double const & time);

  void
  rhs_ppe_nbc_analytical_time_derivative_add(VectorType & dst, double const & time);

  void
  rhs_ppe_nbc_numerical_time_derivative_add(VectorType & dst, VectorType const & src);

  // rhs pressure: Neumann BC convective term
  void
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure: Neumann BC viscous term
  void
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const;

  void
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src, bool const update_preconditioner) const;

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
  local_rhs_ppe_div_term_convective_term_boundary_face(MatrixFree<dim, Number> const & matrix_free,
                                                       VectorType &                    dst,
                                                       VectorType const &              src,
                                                       Range const & face_range) const;

  // body force term
  void
  local_rhs_ppe_div_term_body_forces_boundary_face(MatrixFree<dim, Number> const & matrix_free,
                                                   VectorType &                    dst,
                                                   VectorType const &              src,
                                                   Range const & face_range) const;

  // Neumann boundary condition term

  // body force term
  void
  local_rhs_ppe_nbc_body_force_term_add_boundary_face(MatrixFree<dim, Number> const & matrix_free,
                                                      VectorType &                    dst,
                                                      VectorType const &              src,
                                                      Range const & face_range) const;

  // dg_u/dt term with analytical derivative
  void
  local_rhs_ppe_nbc_analytical_time_derivative_add_boundary_face(
    MatrixFree<dim, Number> const & matrix_free,
    VectorType &                    dst,
    VectorType const &              src,
    Range const &                   face_range) const;

  // dg_u/dt with numerical time derivative
  void
  local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face(
    MatrixFree<dim, Number> const & matrix_free,
    VectorType &                    dst,
    VectorType const &              src,
    Range const &                   face_range) const;

  // convective term
  void
  local_rhs_ppe_nbc_convective_add_boundary_face(MatrixFree<dim, Number> const & matrix_free,
                                                 VectorType &                    dst,
                                                 VectorType const &              src,
                                                 Range const &                   face_range) const;

  // viscous term
  void
  local_rhs_ppe_nbc_viscous_add_boundary_face(MatrixFree<dim, Number> const & matrix_free,
                                              VectorType &                    dst,
                                              VectorType const &              src,
                                              Range const &                   face_range) const;


  /*
   * Viscous step (Helmholtz-like equation).
   */
  std::shared_ptr<PreconditionerBase<Number>> helmholtz_preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> helmholtz_solver;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_DUAL_SPLITTING_H_ \
        */
