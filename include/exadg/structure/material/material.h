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
   * Evaluate 2nd Piola-Kirchhoff stress tensor given the gradient of the displacement field
   * with respect to the reference configuration (not to be confused with the deformation gradient)
   */
  virtual dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  second_piola_kirchhoff_stress(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    unsigned int const                                              cell,
    unsigned int const                                              q) const = 0;

  /*
   * Evaluate directional derivative of the 2nd Piola-Kirchhoff stress tensor given the displacement
   * increment shape function gradient *gradient_increment* with respect to the reference
   * configuration and deformation gradient at the current linearization point *deformation_gradient*
   */
  virtual dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  second_piola_kirchhoff_stress_derivative(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_increment,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & deformation_gradient,
    unsigned int const                                              cell,
    unsigned int const                                              q) const = 0;
};

} // namespace Structure
} // namespace ExaDG

#endif
