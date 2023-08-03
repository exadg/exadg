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

#ifndef INCLUDE_EXADG_STRUCTURE_MATERIAL_MATERIAL_H_
#define INCLUDE_EXADG_STRUCTURE_MATERIAL_MATERIAL_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/structure/material/material_data.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class Material
{
public:
  virtual ~Material()
  {
  }

  /*
   * Evaluate 2nd Piola-Kirchhoff stress tensor given a strain measure, e.g., the engineering
   * strain, Green-Lagrange strain tensor or Cauchy-Green strain tensor
   */
  virtual dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  PK2_stress(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & strain_measure,
             unsigned int const                                              cell,
             unsigned int const                                              q) const = 0;

  /*
   * Evaluate directional derivative of the 2nd Piola-Kirchhoff stress tensor given the displacement
   * shape function gradient Grad_delta and deformation gradient at current linearization point
   * F_lin
   */
  virtual dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  PK2_stress_derivative(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & Grad_delta,
                        dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & F_lin,
                        unsigned int const                                              cell,
                        unsigned int const q) const = 0;
};

} // namespace Structure
} // namespace ExaDG

#endif
