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
#include <exadg/matrix_free/integrators.h>
#include <exadg/structure/material/material_data.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class Material
{
public:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  virtual ~Material()
  {
  }

  /*
   * Total Lagrangian Formulation: evaluate 2nd Piola-Kirchhoff stress tensor given
   * the gradient of the displacement field with respect to the reference configuration
   * (not to be confused with the deformation gradient).
   */
  virtual tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement_cache_level_0_1,
                                unsigned int const cell,
                                unsigned int const q) const
  {
    (void)gradient_displacement_cache_level_0_1;
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage("For a total Lagrangian formulation,"
                                   "overwrite this method in derived class."));

    tensor dummy;
    return dummy;
  }

  /*
   * Total Lagrangian Formulation: evaluate the directional derivative with respect
   * to the displacement of the 2nd Piola-Kirchhoff stress tensor given gradient of
   * the displacement increment with respect to the reference configuration
   * "gradient_increment" and deformation gradient at the current linearization point
   * "deformation_gradient".
   */
  virtual tensor
  second_piola_kirchhoff_stress_displacement_derivative(
    tensor const &     gradient_increment,
    tensor const &     gradient_displacement_cache_level_0_1,
    unsigned int const cell,
    unsigned int const q) const
  {
    (void)gradient_increment;
    (void)gradient_displacement_cache_level_0_1;
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage("For a total Lagrangian formulation, "
                                   "overwrite this method in derived class."));

    tensor dummy;
    return dummy;
  }

  /*
   * Lagrangian formulation with integration in the spatial configuration: provide
   * Kirchhoff stress tau = J * sigma = F * S * F^T given the gradient of the
   * displacement field with respect to the reference configuration
   * (not to be confused with the deformation gradient).
   */
  virtual tensor
  kirchhoff_stress(tensor const &     gradient_displacement_cache_level_0_1,
                   unsigned int const cell,
                   unsigned int const q) const
  {
    (void)gradient_displacement_cache_level_0_1;
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage("For a Lagrangian formulation in spatial domain, "
                                   "overwrite this method in derived class."));

    tensor dummy;
    return dummy;
  }

  /*
   * Lagrangian formulation with integration in the spatial configuration: provide
   * operation J*c:(X), where c is the spatial tangent tensor and X is a symmetric
   * second order tensor.
   */
  virtual tensor
  contract_with_J_times_C(tensor const &     symmetric_gradient_increment,
                          tensor const &     gradient_displacement_cache_level_0_1,
                          unsigned int const cell,
                          unsigned int const q) const
  {
    (void)symmetric_gradient_increment;
    (void)gradient_displacement_cache_level_0_1;
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage("For a Lagrangian formulation in spatial domain, "
                                   "overwrite this method in derived class."));

    tensor dummy;
    return dummy;
  }

  /*
   * Update the linearization data stored for in each integration point via VariableCoefficients
   * given the current linearization vector.
   */
  virtual void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const
  {
    (void)integrator_lin;
    (void)cell;
  }

  virtual scalar
  one_over_J(unsigned int const cell, unsigned int const q) const
  {
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage(
                  "Overwrite this method in derived class to access stored one_over_J."));

    scalar dummy;
    return dummy;
  }

  virtual tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const
  {
    (void)cell;
    (void)q;
    AssertThrow(false,
                dealii::ExcMessage(
                  "Overwrite this method in derived class to access stored deformation gradient."));

    tensor dummy;
    return dummy;
  }
};

} // namespace Structure
} // namespace ExaDG

#endif
