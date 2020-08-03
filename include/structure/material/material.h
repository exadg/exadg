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
  virtual ~Material()
  {
  }

  virtual Tensor<2, dim, VectorizedArray<Number>>
    evaluate_stress(Tensor<2, dim, VectorizedArray<Number>> const & E,
                    unsigned int const                              cell,
                    unsigned int const                              q) const = 0;

  virtual Tensor<2, dim, VectorizedArray<Number>>
    apply_C(Tensor<2, dim, VectorizedArray<Number>> const & E,
            unsigned int const                              cell,
            unsigned int const                              q) const = 0;
};

} // namespace Structure

#endif
