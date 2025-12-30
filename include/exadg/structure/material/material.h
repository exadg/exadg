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

#ifndef EXADG_STRUCTURE_MATERIAL_MATERIAL_H_
#define EXADG_STRUCTURE_MATERIAL_MATERIAL_H_

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
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, dim, Number>                   IntegratorCell;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

  virtual ~Material()
  {
  }

  /*
   * Evaluate 2nd Piola-Kirchhoff stress tensor given the gradient of the displacement field
   * with respect to the reference configuration (not to be confused with the deformation gradient).
   */
  virtual symmetric_tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const = 0;

  /*
   * Evaluate the directional derivative with respect to the displacement of the 2nd Piola-Kirchhoff
   * stress tensor given gradient of the displacment increment with respect to the reference
   * configuration `gradient_increment` and deformation gradient at the current linearization point
   * `deformation_gradient`.
   */
  virtual symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     deformation_gradient,
                                                        unsigned int const cell,
                                                        unsigned int const q) const = 0;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_MATERIAL_MATERIAL_H_ */
