/*
 * boundary_conditions.h
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
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
                                                        integrator.get_face_index(),
                                                        q,
                                                        integrator.quadrature_formula_index());
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
