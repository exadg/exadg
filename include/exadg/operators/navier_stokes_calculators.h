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

#ifndef INCLUDE_EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_
#define INCLUDE_EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
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
  void
  compute_divergence(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};

template<int dim, typename Number>
class ShearRateCalculator
{
private:
  typedef ShearRateCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetrictensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

public:
  ShearRateCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_u_in,
             unsigned int const                      dof_index_u_scalar_in,
             unsigned int const                      quad_index_in);

  void
  compute_shear_rate(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

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
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in);

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int                            dof_index;
  unsigned int                            quad_index;
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
             unsigned int const                      dof_index_u_in,
             unsigned int const                      dof_index_u_scalar_in,
             unsigned int const                      quad_index_in);

  void
  compute(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

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
             unsigned int const                      dof_index_u_in,
             unsigned int const                      dof_index_u_scalar_in,
             unsigned int const                      quad_index_in,
             bool const                              compressible_flow);

  void
  compute(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
  bool         compressible_flow;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_NAVIER_STOKES_CALCULATORS_H_ \
        */
