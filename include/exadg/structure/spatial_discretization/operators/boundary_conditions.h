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
#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

/*
 * This function calculates the Neumann boundary value.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  calculate_neumann_value(unsigned int const                             q,
                          FaceIntegrator<dim, dim, Number> const &       integrator,
                          BoundaryType const &                           boundary_type,
                          types::boundary_id const                       boundary_id,
                          std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
                          double const &                                 time)
{
  Tensor<1, dim, VectorizedArray<Number>> traction;

  if(boundary_type == BoundaryType::Neumann)
  {
    auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    traction = FunctionEvaluator<1, dim, Number>::value(bc, q_points, time);
  }
  else if(boundary_type == BoundaryType::NeumannMortar)
  {
    auto bc = boundary_descriptor->neumann_mortar_bc.find(boundary_id)->second;

    traction = FunctionEvaluator<1, dim, Number>::value(bc,
                                                        integrator.get_current_cell_index(),
                                                        q,
                                                        integrator.get_quadrature_index());
  }
  else
  {
    // do nothing

    AssertThrow(boundary_type == BoundaryType::Dirichlet ||
                  boundary_type == BoundaryType::DirichletMortar,
                ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return traction;
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_ */
