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

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  add_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] + 1.0;
  return gradient;
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  subtract_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] - 1.0;
  return gradient;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_F(const dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & H)
{
  return add_identity(H);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_E(const dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & F)
{
  return 0.5 * subtract_identity(transpose(F) * F);
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
