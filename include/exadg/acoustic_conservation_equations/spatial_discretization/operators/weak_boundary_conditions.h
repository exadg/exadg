/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_

#include <exadg/acoustic_conservation_equations/user_interface/boundary_descriptor.h>
#include <exadg/functions_and_boundary_conditions/boundary_face_integrator_base.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace Acoustics
{
// compute exterior pressure values for different boundary types
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  calculate_exterior_value_pressure(unsigned int const                     q,
                                    FaceIntegrator<dim, 1, Number> const & integrator_m,
                                    BoundaryTypeP const &                  boundary_type,
                                    dealii::types::boundary_id const       boundary_id,
                                    BoundaryDescriptorP<dim> const &       boundary_descriptor,
                                    Number const                           time)
{
  if(boundary_type == BoundaryTypeP::Dirichlet)
  {
    auto const g = FunctionEvaluator<0, dim, Number>::value(
      *boundary_descriptor.dirichlet_bc.find(boundary_id)->second,
      integrator_m.quadrature_point(q),
      time);
    return -integrator_m.get_value(q) + Number{2.0} * g;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }
}

// compute exterior velocity values for different boundary types
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
  calculate_exterior_value_velocity(unsigned int const                       q,
                                    FaceIntegrator<dim, dim, Number> const & integrator_m,
                                    BoundaryTypeU const &                    boundary_type,
                                    dealii::types::boundary_id const         boundary_id,
                                    BoundaryDescriptorU<dim> const &         boundary_descriptor,
                                    Number const                             time)
{
  (void)boundary_id;
  (void)boundary_descriptor;
  (void)time;

  if(boundary_type == BoundaryTypeU::Neumann)
  {
    return integrator_m.get_value(q);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }
}


/**
 * Class to access values of pressure at boundaries similar to FaceIntegrators.
 */
template<int dim, typename Number>
class BoundaryFaceIntegratorP : public BoundaryFaceIntegratorBase<BoundaryDescriptorP<dim>, Number>
{
  using FaceIntegratorP = FaceIntegrator<dim, 1, Number>;

public:
  BoundaryFaceIntegratorP(FaceIntegratorP const &          integrator_m_in,
                          BoundaryDescriptorP<dim> const & boundary_descriptor_in)
    : BoundaryFaceIntegratorBase<BoundaryDescriptorP<dim>, Number>(
        integrator_m_in.get_matrix_free(),
        boundary_descriptor_in),
      integrator_m(integrator_m_in)
  {
  }

  inline DEAL_II_ALWAYS_INLINE //
    typename FaceIntegratorP::value_type
    get_value(unsigned int const q) const
  {
    return calculate_exterior_value_pressure(q,
                                             integrator_m,
                                             this->boundary_type,
                                             this->boundary_id,
                                             this->boundary_descriptor,
                                             this->evaluation_time);
  }

private:
  FaceIntegratorP const & integrator_m;
};

/**
 * Same as above for the velocity.
 */
template<int dim, typename Number>
class BoundaryFaceIntegratorU : public BoundaryFaceIntegratorBase<BoundaryDescriptorU<dim>, Number>
{
  using FaceIntegratorU = FaceIntegrator<dim, dim, Number>;

public:
  BoundaryFaceIntegratorU(FaceIntegratorU const &          integrator_m_in,
                          BoundaryDescriptorU<dim> const & boundary_descriptor_in)
    : BoundaryFaceIntegratorBase<BoundaryDescriptorU<dim>, Number>(
        integrator_m_in.get_matrix_free(),
        boundary_descriptor_in),
      integrator_m(integrator_m_in)
  {
  }

  inline DEAL_II_ALWAYS_INLINE //
    typename FaceIntegratorU::value_type
    get_value(unsigned int const q) const
  {
    return calculate_exterior_value_velocity(q,
                                             integrator_m,
                                             this->boundary_type,
                                             this->boundary_id,
                                             this->boundary_descriptor,
                                             this->evaluation_time);
  }

private:
  FaceIntegratorU const & integrator_m;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /*EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_*/
