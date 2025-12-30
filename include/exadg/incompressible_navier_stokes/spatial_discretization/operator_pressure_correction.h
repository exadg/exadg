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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>

namespace ExaDG
{
namespace IncNS
{
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
  OperatorPressureCorrection(
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
  virtual ~OperatorPressureCorrection();

  /*
   * Initializes the inverse pressure mass matrix operator needed for the pressure correction
   * scheme, as well as the pressure mass operator needed in the ALE case only (where the mass
   * operator may be evaluated at different times depending on the specific ALE formulation chosen).
   */
  void
  setup_derived() final;

  void
  update_after_grid_motion(bool const update_matrix_free) final;

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
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PRESSURE_CORRECTION_H_ \
        */
