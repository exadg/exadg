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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace IncNS
{
namespace Operators
{
template<int dim, typename Number>
class GradientKernel
{
private:
  typedef CellIntegrator<dim, 1, Number> CellIntegratorP;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells       = dealii::update_JxW_values | dealii::update_gradients;
    flags.inner_faces = dealii::update_JxW_values | dealii::update_normal_vectors;
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_normal_vectors;

    return flags;
  }

  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(scalar const & value_m, scalar const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral for
   * weak formulation (performing integration-by-parts)
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux_weak(CellIntegratorP & pressure, unsigned int const q) const
  {
    // minus sign due to integration by parts
    return -pressure.get_value(q);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral for
   * strong formulation (no integration-by-parts, or integration-by-parts performed twice)
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_strong(CellIntegratorP & pressure, unsigned int const q) const
  {
    return pressure.get_gradient(q);
  }
};
} // namespace Operators

template<int dim>
struct GradientOperatorData
{
  GradientOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true),
      formulation(FormulationPressureGradientTerm::Weak)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  FormulationPressureGradientTerm formulation;

  std::shared_ptr<BoundaryDescriptorP<dim> const> bc;
};

template<int dim, typename Number>
class GradientOperator
{
public:
  typedef GradientOperator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorP;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorP;

  GradientOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             GradientOperatorData<dim> const &       data_in);

  void
  set_scaling_factor_pressure(double const & scaling_factor);

  GradientOperatorData<dim> const &
  get_operator_data() const;

  // homogeneous operator
  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

  // inhomogeneous operator
  void
  rhs(VectorType & dst, Number const evaluation_time) const;

  void
  rhs_bc_from_dof_vector(VectorType & dst, VectorType const & pressure_bc) const;

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const;

  // full operator, i.e., homogeneous and inhomogeneous contributions
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

  void
  evaluate_bc_from_dof_vector(VectorType &       dst,
                              VectorType const & src,
                              VectorType const & pressure_bc) const;

private:
  void
  do_cell_integral_weak(CellIntegratorP & pressure, CellIntegratorU & velocity) const;

  void
  do_cell_integral_strong(CellIntegratorP & pressure, CellIntegratorU & velocity) const;

  void
  do_face_integral(FaceIntegratorP & pressure_m,
                   FaceIntegratorP & pressure_p,
                   FaceIntegratorU & velocity_m,
                   FaceIntegratorU & velocity_p) const;

  void
  do_boundary_integral(FaceIntegratorP &                  pressure,
                       FaceIntegratorU &                  velocity,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  void
  do_boundary_integral_from_dof_vector(FaceIntegratorP &                  pressure,
                                       FaceIntegratorP &                  pressure_exterior,
                                       FaceIntegratorU &                  velocity,
                                       OperatorType const &               operator_type,
                                       dealii::types::boundary_id const & boundary_id) const;

  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  void
  face_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           face_range) const;

  void
  boundary_face_loop_hom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                            dst,
                                  VectorType const &                      src,
                                  Range const &                           face_range) const;

  void
  boundary_face_loop_full_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                            dst,
                                   VectorType const &                      src,
                                   Range const &                           face_range) const;

  void
  cell_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           cell_range) const;

  void
  face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           face_range) const;

  void
  boundary_face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           face_range) const;

  void
  boundary_face_loop_inhom_operator_bc_from_dof_vector(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  void
  boundary_face_loop_full_operator_bc_from_dof_vector(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           face_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  GradientOperatorData<dim> data;

  mutable double time;

  // if the continuity equation of the incompressible Navier-Stokes
  // equations is scaled by a constant factor, the system of equations
  // is solved for a modified pressure p^* = 1/scaling_factor * p. Hence,
  // when applying the gradient operator to this modified pressure we have
  // to make sure that we also apply the correct boundary conditions for p^*,
  // i.e., g_p^* = 1/scaling_factor * g_p
  double inverse_scaling_factor_pressure;

  Operators::GradientKernel<dim, Number> kernel;

  // needed if pressure Dirichlet boundary condition is evaluated from dof vector
  mutable VectorType const * pressure_bc;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_ \
        */
