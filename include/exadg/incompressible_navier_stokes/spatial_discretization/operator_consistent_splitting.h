/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_H_

// base class
#include <exadg/incompressible_navier_stokes/spatial_discretization/curl_compute.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number = double>
class OperatorConsistentSplitting : public OperatorProjectionMethods<dim, Number>
{
private:
  typedef SpatialOperatorBase<dim, Number>         Base;
  typedef OperatorProjectionMethods<dim, Number>   ProjectionBase;
  typedef OperatorConsistentSplitting<dim, Number> This;

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
  OperatorConsistentSplitting(
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
  virtual ~OperatorConsistentSplitting();

  /*
   * Pressure Poisson equation.
   */
  // Leray projection
  void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const;

  // rhs pressure: divergence of convective term
  void
  apply_convective_divergence_term(VectorType & dst, VectorType const & src) const;

  // rhs pressure Poisson equation: velocity divergence term: body force term
  void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & time) const;

  // rhs pressure Poisson equation: Neumann BC viscous term and numerical time derivative term
  void
  rhs_ppe_nbc_add(VectorType &       dst,
                  VectorType const & src,
                  double const &     time,
                  Number const       gamma_dt) const;

  /*
   * Viscous step.
   */
  void
  apply_helmholtz_operator(VectorType & dst, VectorType const & src) const;

private:
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

  /*
   * Right-hand side of the PPE
   */
  // The bdf constant for the time derivative divded by the timestep size
  mutable Number gamma0_dt;

  // body force term
  void
  local_rhs_ppe_div_term_body_forces_cell(dealii::MatrixFree<dim, Number> const & matrix_free,
                                          VectorType &                            dst,
                                          VectorType const &                      src,
                                          Range const &                           cell_range) const;

  void
  local_rhs_ppe_div_term_body_forces_inner_face(dealii::MatrixFree<dim, Number> const & matrix_free,
                                                VectorType &                            dst,
                                                VectorType const &                      src,
                                                Range const & face_range) const;

  void
  local_rhs_ppe_div_term_body_forces_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  // divergence of convective term
  void
  local_rhs_ppe_div_term_convective_cell(dealii::MatrixFree<dim, Number> const & matrix_free,
                                         VectorType &                            dst,
                                         VectorType const &                      src,
                                         Range const &                           cell_range) const;

  void
  local_rhs_ppe_div_term_convective_inner_face(dealii::MatrixFree<dim, Number> const & matrix_free,
                                               VectorType &                            dst,
                                               VectorType const &                      src,
                                               Range const & face_range) const;

  void
  local_rhs_ppe_div_term_convective_boundary_face(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;


  /*
   * Neumann boundary condition term
   */
  // viscous term and Numerical time derivative of the Dirichlet data dg_u/dt using suitable BDF
  void
  local_rhs_ppe_nbc_add_boundary_face(dealii::MatrixFree<dim, Number> const & data,
                                      VectorType &                            dst,
                                      VectorType const &                      src,
                                      Range const &                           face_range) const;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_H_ \
        */
