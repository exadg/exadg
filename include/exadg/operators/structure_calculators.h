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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_
#define EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>

namespace ExaDG
{
/*
 * Calculator for the Jacobian of the vector field u(x) defined as
 *
 * J := det(F) with F := I + grad(u),
 *
 * where F is the deformation gradient tensor and grad(u) is the gradient of the vector field.
 *
 */
template<int dim, typename Number>
class DisplacementJacobianCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DisplacementJacobianCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  DisplacementJacobianCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the Jacobian of the vector field.
   */
  void
  compute_projection_rhs(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst_scalar_valued,
            VectorType const &                            src_vector_valued,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};

/*
 * Calculator for the maximum principal stress being the largest absolute Eigenvalue of the Cauchy
 * stress tensor
 *
 * sigma_I := max(|lambda_i|) for i = 1,...,dim
 *
 * with lambda_i being the Eigenvalues of the Cauchy stress tensor sigma,
 *
 * sigma = (1/det(F)) * F * S * F^T
 *
 * where F is the deformation gradient tensor and S is the second Piola-Kirchhoff stress tensor.
 *
 */
template<int dim, typename Number>
class MaxPrincipalStressCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef MaxPrincipalStressCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  MaxPrincipalStressCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free_in,
             unsigned int const                                     dof_index_vector_in,
             unsigned int const                                     dof_index_scalar_in,
             unsigned int const                                     quad_index_in,
             Structure::ElasticityOperatorBase<dim, Number> const & elasticity_operator_base_in);

  /*
   * Compute the right-hand side of an L2 projection of the maximum principal stress.
   */
  void
  compute_projection_rhs(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst_scalar_valued,
            VectorType const &                            src_vector_valued,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;

  // Pointer to the underlying elasticity operator to compute stresses dependent on the material
  // model.
  dealii::ObserverPointer<Structure::ElasticityOperatorBase<dim, Number> const>
    elasticity_operator_base;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_ */
