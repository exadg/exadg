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

#ifndef EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_
#define EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h>
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
/*
 * Calculator for the divergence of a vector field u(x) defined as the scalar quantity

 * div(u) := sum_i=1,...,dim { d u(i) / d x(i) }.
 */
template<int dim, typename Number>
class DivergenceCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DivergenceCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

  typedef dealii::VectorizedArray<Number> scalar;

  DivergenceCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the divergence of the vector field.
   */
  void
  compute_projection_rhs(VectorType &       dst_scalar_valued,
                         VectorType const & src_vector_valued) const;

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
 * Calculator for the shear rate according to
 * [Galdi et al., 2008, Hemodynamical Flows: Modeling, Analysis and Simulation]
 * of a vector field u(x) defined as the scalar quantity
 *
 * sqrt(2 * trace(sym_grad(u)^2)) = sqrt(2 * sym_grad(u) : sym_grad(u))
 * where
 * sym_grad(u) := 0.5 * (grad(u) + grad(u)^T)
 *
 * is the symmetric part of the vector field's gradient.
 */
template<int dim, typename Number>
class ShearRateCalculator
{
private:
  typedef ShearRateCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

public:
  ShearRateCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the shear rate of the vector field.
   */
  void
  compute_projection_rhs(VectorType &       dst_scalar_valued,
                         VectorType const & src_vector_valued) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst_scalar_valued,
            VectorType const &                      src_vector_valued,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};

/*
 * Calculator for the vorticity of a vector field u(x) defined as
 *
 * omega := curl(u)
 *
 * 2D : omega = - d u(1) / d x(2) + d u(2) / d x(1)
 *
 * 3D : omega(1) = d u(3) / d x(2) - d u(2) / d x(3)
 *      omega(2) = d u(1) / d x(3) - d u(3) / d x(1)
 *      omega(3) = d u(2) / d x(1) - d u(1) / d x(2)
 *
 * which is scalar-valued in 2D and vector-valued in 3D. Therefore, we always write into a
 * vector-valued destination vector with `dim` components, filling only the first component in 2D.
 * The second component in the solution vector in the 2D case corresponds to projecting the zero
 * function into the finite element space.
 */
template<int dim, typename Number>
class VorticityCalculator
{
public:
  static unsigned int const number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VorticityCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  VorticityCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the vorticity of the vector field.
   */
  void
  compute_projection_rhs(VectorType &       dst_vector_valued,
                         VectorType const & src_vector_valued) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst_vector_valued,
            VectorType const &                            src_vector_valued,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int                            dof_index_vector;
  unsigned int                            quad_index;
};

/*
 * Calculator to project the viscosity coefficients stored in the `ViscousKernel` into the finite
 * element space. Note that the viscosity is *not updated* but only accessed.
 */
template<int dim, typename Number>
class ViscosityCalculator
{
private:
  typedef ViscosityCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, 1, Number> CellIntegratorScalar;

public:
  ViscosityCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const &              matrix_free_in,
             unsigned int const                                   dof_index_scalar_in,
             unsigned int const                                   quad_index_in,
             IncNS::Operators::ViscousKernel<dim, Number> const & viscous_kernel_in);

  /*
   * Compute the right-hand side of an L2 projection of the stored viscosity.
   */
  void
  compute_projection_rhs(VectorType & dst_scalar_valued) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst_scalar_valued,
            VectorType const &                      src_scalar_valued,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_scalar;
  unsigned int quad_index;

  IncNS::Operators::ViscousKernel<dim, Number> const * viscous_kernel;
};

template<int dim, typename Number>
class MagnitudeCalculator
{
private:
  typedef MagnitudeCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   IntegratorScalar;

public:
  MagnitudeCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the magnitude of the vector field.
   */
  void
  compute_projection_rhs(VectorType &       dst_scalar_valued,
                         VectorType const & src_vector_valued) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst_scalar_valued,
            VectorType const &                      src_vector_valued,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};

/*
 * Calculator to project the Q-criterion into the finite element space.
 */
template<int dim, typename Number>
class QCriterionCalculator
{
private:
  typedef QCriterionCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

public:
  QCriterionCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in,
             bool const                              compressible_flow);

  /*
   * Compute the right-hand side of an L2 projection of the QCriterion of the vector field.
   */
  void
  compute_projection_rhs(VectorType &       dst_scalar_valued,
                         VectorType const & src_vector_valued) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst_scalar_valued,
            VectorType const &                      src_vector_valued,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
  bool         compressible_flow;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_ */
