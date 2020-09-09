/*
 * continuum_mechanics.h
 *
 *  Created on: 03.05.2020
 *      Author: fehn
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
using namespace dealii;

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<Number>>
    add_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] + 1.0;
  return gradient;
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<Number>>
    subtract_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] - 1.0;
  return gradient;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, dim, VectorizedArray<Number>>
  get_F(const Tensor<2, dim, VectorizedArray<Number>> & H)
{
  return add_identity(H);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, dim, VectorizedArray<Number>>
  get_E(const Tensor<2, dim, VectorizedArray<Number>> & F)
{
  return 0.5 * subtract_identity(transpose(F) * F);
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
