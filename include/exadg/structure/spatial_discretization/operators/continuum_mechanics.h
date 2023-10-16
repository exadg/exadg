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
  void
  add_scaled_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & tmp,
                      dealii::VectorizedArray<Number> const &                   factor)
{
  for(unsigned int i = 0; i < dim; i++)
  {
    tmp[i][i] = tmp[i][i] + factor;
  }
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  add_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & tmp)
{
  auto result = tmp;
  for(unsigned int i = 0; i < dim; i++)
  {
    result[i][i] = result[i][i] + 1.0;
  }
  return result;
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  subtract_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & tmp)
{
  auto result = tmp;
  for(unsigned int i = 0; i < dim; i++)
  {
    result[i][i] = result[i][i] - 1.0;
  }
  return result;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_F(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  return add_identity(gradient_displacement);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_E(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & F)
{
  return (0.5 * subtract_identity(transpose(F) * F));
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_identity()
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> I;
  return add_identity(I);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  void
  get_modified_F_J(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> &       F,
    dealii::VectorizedArray<Number> &                               J,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    unsigned int const                                              check_type,
    bool const                                                      compute_J)
{
  F = add_identity(gradient_displacement);

  if(compute_J)
  {
    J = determinant(F);
  }

  if(check_type > 0)
  {
    if(checktype == 1)
    {
  	  AssertThrow(false, dealii::ExcMessage("This check_type is not defined."));
    }
	else
    {
	  AssertThrow(false, dealii::ExcMessage("This check_type is not defined."));
    }
  }
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
