/*
 * divergence_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_


#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/mapping_flags.h"
#include "../../user_interface/input_parameters.h"
#include "weak_boundary_conditions.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
template<int dim, typename Number>
class DivergenceKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells          = update_JxW_values | update_gradients;
    flags.inner_faces    = update_JxW_values | update_normal_vectors;
    flags.boundary_faces = update_JxW_values | update_quadrature_points | update_normal_vectors;

    return flags;
  }

  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & value_m, vector const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral for
   * weak formulation (performing integration-by-parts)
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_weak(CellIntegratorU & velocity, unsigned int const q) const
  {
    // minus sign due to integration by parts
    return -velocity.get_value(q);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral for
   * strong formulation (no integration-by-parts, or integration-by-parts performed twice)
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux_strong(CellIntegratorU & velocity, unsigned int const q) const
  {
    return velocity.get_divergence(q);
  }
};
} // namespace Operators

template<int dim>
struct DivergenceOperatorData
{
  DivergenceOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, typename Number>
class DivergenceOperator
{
public:
  typedef DivergenceOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorP;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorP;

  DivergenceOperator();

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free_in,
         DivergenceOperatorData<dim> const & data_in);

  // homogeneous operator
  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

  // inhomogeneous operator
  void
  rhs(VectorType & dst, Number const evaluation_time) const;

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const;

  // full operator, i.e., homogeneous and inhomogeneous contributions
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

private:
  void
  do_cell_integral_weak(CellIntegratorP & pressure, CellIntegratorU & velocity) const;

  void
  do_cell_integral_strong(CellIntegratorP & pressure, CellIntegratorU & velocity) const;

  void
  do_face_integral(FaceIntegratorU & velocity_m,
                   FaceIntegratorU & velocity_p,
                   FaceIntegratorP & pressure_m,
                   FaceIntegratorP & pressure_p) const;

  void
  do_boundary_integral(FaceIntegratorU &          velocity,
                       FaceIntegratorP &          pressure,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const;

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const;

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const;

  void
  cell_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   cell_range) const;

  void
  face_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   face_range) const;

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const &               matrix_free,
                                    VectorType &                                  dst,
                                    VectorType const &                            src,
                                    std::pair<unsigned int, unsigned int> const & face_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  DivergenceOperatorData<dim> data;

  mutable double time;

  Operators::DivergenceKernel<dim, Number> kernel;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_ \
        */
