/*
 * rhs_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_

// deal.II
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../user_interface/boundary_descriptor.h"

#include "../../../operators/mapping_flags.h"

using namespace dealii;

namespace Structure
{
template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0), degree(1), do_rhs(false)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  unsigned int degree;

  bool                           do_rhs;
  std::shared_ptr<Function<dim>> rhs;

  std::shared_ptr<BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class RHSOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef RHSOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  /*
   * Constructor.
   */
  RHSOperator();

  /*
   * Initialization.
   */
  void
  reinit(MatrixFree<dim, Number> const &   mf_data,
         DoFHandler<dim> const &           dof_handler,
         AffineConstraints<double> const & constraint_matrix,
         Mapping<dim> const &              mapping,
         RHSOperatorData<dim> const &      operator_data_in);

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_JxW_values | update_quadrature_points;

    flags.boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

  void
  evaluate_add_nbc(VectorType & dst, double const evaluation_time) const;

private:
  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  void
  cell_loop_empty(MatrixFree<dim, Number> const & data,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   cell_range) const;

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  void
  boundary_loop(MatrixFree<dim, Number> const & data,
                VectorType &                    dst,
                VectorType const &              src,
                Range const &                   cell_range) const;

  void
  do_boundary(VectorType & dst) const;

  MatrixFree<dim, Number> const * data;

  DoFHandler<dim> const * dof_handler;

  AffineConstraints<double> const * constraint_matrix;

  Mapping<dim> const * mapping;

  RHSOperatorData<dim> operator_data;

  double mutable eval_time;
};

} // namespace Structure

#endif
