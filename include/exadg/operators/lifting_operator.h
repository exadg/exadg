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

#ifndef INCLUDE_OPERATORS_LIFTING_OPERATOR
#define INCLUDE_OPERATORS_LIFTING_OPERATOR

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace Operators
{
template<int dim>
struct LiftingKernelData
{
 // Boundary conditions for lifting
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_conditions;
};

template<int dim, typename Number, int n_components = 1>
class LiftingKernel
{
private:
  typedef CellIntegrator<dim, 1, Number> IntegratorCellScalar;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFaceScalar;

  typedef CellIntegrator<dim, dim, Number> IntegratorCellVector;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFaceVector;

  typedef dealii::VectorizedArray<Number>   scalar;
  typedef dealii::Tensor<1, dim, scalar> vector;

public:
  void
  reinit(LiftingKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values |
                  dealii::update_quadrature_points; // q-points due to rhs function f

    flags.inner_faces = dealii::update_JxW_values |
                          dealii::update_normal_vectors |
                          dealii::update_quadrature_points;

    flags.boundary_faces = dealii::update_JxW_values |
                             dealii::update_normal_vectors |
                             dealii::update_quadrature_points;

    return flags;
  }

  inline DEAL_II_ALWAYS_INLINE
  vector
  get_face_integral(scalar const & val_m,
                    scalar const & val_p,
                    vector const & normal) const
  {
    return 0.5 * (val_p - val_m) * normal;
  }

private:
  mutable LiftingKernelData<dim> data;
};

} // namespace Operators


template<int dim>
struct LiftingOperatorData
{
  LiftingOperatorData() : dof_index_scalar(0), dof_index_vector(0), quad_index(0)
  {
  }

  unsigned int dof_index_scalar;
  unsigned int dof_index_vector;
  unsigned int quad_index;

  Operators::LiftingKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class LiftingOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef LiftingOperator<dim, Number, n_components> This;

  typedef CellIntegrator<dim, 1, Number> IntegratorCellScalar;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFaceScalar;

  typedef CellIntegrator<dim, dim, Number> IntegratorCellVector;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFaceVector;

  typedef dealii::VectorizedArray<Number>   scalar;
  typedef dealii::Tensor<1, dim, scalar> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * Constructor.
   */
  LiftingOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             LiftingOperatorData<dim> const &            data);

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

  void
  evaluate_lifting_operator(VectorType const & solution,
                            VectorType &       lifting_term,
                            double const       evaluation_time) const;

  void
  apply_inverse_mass_matrix(VectorType & dst,
                            VectorType const & src) const;
private:
  void
  do_cell_integral(IntegratorCellVector & integrator) const;

  void
  face_loop_lifting(dealii::MatrixFree<dim, Number> const & matrix_free,
                    VectorType &                            dst,
                    VectorType const &                      src,
                    Range const &                           face_range) const;

  void
  boundary_face_loop_lifting(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &                            dst,
                             VectorType const &                      src,
                             Range const &                           face_range) const;

  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  LiftingOperatorData<dim> data;

  mutable double time;

  Operators::LiftingKernel<dim, Number, n_components> kernel;
};

} // namespace ExaDG

#endif
