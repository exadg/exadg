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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_DUAL_SPLITTING_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_DUAL_SPLITTING_H_

// base class
#include <exadg/incompressible_navier_stokes/spatial_discretization/curl_compute.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number = double>
class OperatorDualSplitting : public OperatorProjectionMethods<dim, Number>
{
private:
  typedef SpatialOperatorBase<dim, Number>       Base;
  typedef OperatorProjectionMethods<dim, Number> ProjectionBase;
  typedef OperatorDualSplitting<dim, Number>     This;

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
  OperatorDualSplitting(std::shared_ptr<Grid<dim> const>                  grid,
                        std::shared_ptr<GridMotionInterface<dim, Number>> grid_motion,
                        std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor,
                        std::shared_ptr<FieldFunctions<dim> const>        field_functions,
                        Parameters const &                                parameters,
                        std::string const &                               field,
                        MPI_Comm const &                                  mpi_comm);

  /*
   * Destructor.
   */
  virtual ~OperatorDualSplitting();

  void
  setup_solvers(double const & scaling_factor_mass, VectorType const & velocity);

  /*
   * Pressure Poisson equation.
   */

  // rhs pressure: velocity divergence
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  void
  rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                            VectorType const & velocity) const;

  // rhs pressure Poisson equation: velocity divergence term: body force term
  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & time);

  // rhs pressure Poisson equation: velocity divergence term: convective term
  void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure Poisson equation: Neumann BC body force term
  void
  rhs_ppe_nbc_body_force_term_add(VectorType & dst, double const & time);

  // rhs pressure Poisson equation: Neumann BC numerical time derivative term
  void
  rhs_ppe_nbc_numerical_time_derivative_add(VectorType & dst, VectorType const & src);

  // rhs pressure Poisson equation: Neumann BC convective term
  void
  rhs_ppe_nbc_convective_add(VectorType & dst, VectorType const & src) const;

  // rhs pressure Poisson equation: Neumann BC viscous term
  void
  rhs_ppe_nbc_viscous_add(VectorType & dst, VectorType const & src) const;

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
                double const &     scaling_factor_mass);

  /*
   * Fill a DoF vector with velocity Dirichlet values on Dirichlet boundaries.
   *
   * Note that this function only works as long as one uses a nodal dealii::FE_DGQ element with
   * Gauss-Lobatto points. Otherwise, the quadrature formula used in this function does not match
   * the nodes of the element, and the values injected by this function into the DoF vector are not
   * the degrees of freedom of the underlying finite element space.
   *
   * TODO: remove the last parameter
   */
  void
  interpolate_velocity_dirichlet_bc(VectorType &   dst,
                                    double const & time,
                                    bool const     use_dirichlet_cached_bc_data);

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

  // rhs PPE: velocity divergence term

  // convective term
  void
  local_rhs_ppe_div_term_convective_term_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // body force term
  void
  local_rhs_ppe_div_term_body_forces_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // Neumann boundary condition term

  // dg_u/dt with numerical time derivative
  void
  local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // body force term
  void
  local_rhs_ppe_nbc_body_force_term_add_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // convective term
  void
  local_rhs_ppe_nbc_convective_add_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // viscous term
  void
  local_rhs_ppe_nbc_viscous_add_boundary_face(dealii::MatrixFree<dim, Number> const & matrix_free,
                                              VectorType &                            dst,
                                              VectorType const &                      src,
                                              Range const & face_range) const;

  void
  local_interpolate_velocity_dirichlet_bc_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;


  /*
   * Viscous step (Helmholtz-like equation).
   */
  std::shared_ptr<PreconditionerBase<Number>> helmholtz_preconditioner;

  std::shared_ptr<Krylov::SolverBase<VectorType>> helmholtz_solver;

  // TODO: remove this parameter: currently needed for FSI
  bool use_dirichlet_cached_bc_data;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_DUAL_SPLITTING_H_ \
        */
