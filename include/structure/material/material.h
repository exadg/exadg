/*
 * material.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_MATERIAL_MATERIAL_H_
#define INCLUDE_STRUCTURE_MATERIAL_MATERIAL_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include "material_data.h"

using namespace dealii;

namespace Structure
{
template<int dim, typename Number>
class Material
{
public:
  static const int s = dim == 2 ? 3 : 6;

  virtual ~Material()
  {
  }

  virtual void
  reinit(const Tensor<1, s, VectorizedArray<Number>> & deformation) const = 0;

  virtual Tensor<1, s, VectorizedArray<Number>>
  get_S() const = 0;

  virtual const Tensor<2, s, VectorizedArray<Number>> &
  get_dSdE() const = 0;
};

} // namespace Structure

#endif
