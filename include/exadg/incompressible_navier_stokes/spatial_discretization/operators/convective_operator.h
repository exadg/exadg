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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace IncNS
{
namespace Operators
{
struct ConvectiveKernelData
{
  ConvectiveKernelData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      temporal_treatment(TreatmentOfConvectiveTerm::Implicit),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror),
      ale(false)
  {
  }

  FormulationConvectiveTerm formulation;

  TreatmentOfConvectiveTerm temporal_treatment;

  double upwind_factor;

  bool use_outflow_bc;

  TypeDirichletBCs type_dirichlet_bc;

  bool ale;
};

template<int dim, typename Number>
class ConvectiveKernel
{
public:
  ConvectiveKernel(){};


private:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         ConvectiveKernelData const &            data,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index_linearized,
         bool const                              use_own_velocity_storage)
  {
    this->data = data;

    // integrators for linearized problem
    integrator_velocity =
      std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index_linearized);
    integrator_velocity_m =
      std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index_linearized);
    integrator_velocity_p =
      std::make_shared<IntegratorFace>(matrix_free, false, dof_index, quad_index_linearized);

    if(data.ale)
    {
      integrator_grid_velocity =
        std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index_linearized);
      integrator_grid_velocity_face =
        std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index_linearized);
    }

    if(use_own_velocity_storage)
    {
      velocity.reset();
      matrix_free.initialize_dof_vector(velocity.own(), dof_index);
    }

    if(data.ale)
    {
      matrix_free.initialize_dof_vector(grid_velocity.own(), dof_index);

      // grid velocity vector needs to be ghosted
      grid_velocity->update_ghost_values();

      AssertThrow(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation,
                  dealii::ExcMessage(
                    "ALE formulation can only be used in combination with ConvectiveFormulation"));
    }
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells          = dealii::update_JxW_values | dealii::update_gradients;
    flags.inner_faces    = dealii::update_JxW_values | dealii::update_normal_vectors;
    flags.boundary_faces = dealii::update_JxW_values | dealii::update_normal_vectors;

    return flags;
  }

  /*
   * IntegratorFlags valid for both the nonlinear convective operator and the linearized convective
   * operator
   */
  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flags.cell_evaluate  = dealii::EvaluationFlags::values;
      flags.cell_integrate = dealii::EvaluationFlags::gradients;

      flags.face_evaluate  = dealii::EvaluationFlags::values;
      flags.face_integrate = dealii::EvaluationFlags::values;
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flags.cell_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
      flags.cell_integrate = dealii::EvaluationFlags::values;

      flags.face_evaluate  = dealii::EvaluationFlags::values;
      flags.face_integrate = dealii::EvaluationFlags::values;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return flags;
  }

  ConvectiveKernelData const &
  get_data() const
  {
    return this->data;
  }

  VectorType const &
  get_velocity() const
  {
    return *velocity;
  }

  void
  set_velocity_copy(VectorType const & src)
  {
    velocity.own() = src;
    velocity->update_ghost_values();
  }

  void
  set_grid_velocity_ptr(VectorType const & src)
  {
    grid_velocity.reset(src);
    grid_velocity->update_ghost_values();
  }

  VectorType const &
  get_grid_velocity() const
  {
    return *grid_velocity;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_velocity_cell(unsigned int const q) const
  {
    return integrator_velocity->get_value(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_velocity_gradient_cell(unsigned int const q) const
  {
    return integrator_velocity->get_gradient(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_velocity_m(unsigned int const q) const
  {
    return integrator_velocity_m->get_value(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_velocity_p(unsigned int const q) const
  {
    return integrator_velocity_p->get_value(q);
  }

  // grid velocity cell
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity_cell(unsigned int const q) const
  {
    return integrator_grid_velocity->get_value(q);
  }

  // grid velocity face (the grid velocity is continuous
  // so that we need only one function instead of minus and
  // plus functions)
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity_face(unsigned int const q) const
  {
    return integrator_grid_velocity_face->get_value(q);
  }

  // linearized operator
  void
  reinit_cell(unsigned int const cell) const
  {
    integrator_velocity->reinit(cell);

    if(data.ale)
      integrator_grid_velocity->reinit(cell);

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      integrator_velocity->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      integrator_velocity->gather_evaluate(*velocity,
                                           dealii::EvaluationFlags::values |
                                             dealii::EvaluationFlags::gradients);

      if(data.ale)
        integrator_grid_velocity->gather_evaluate(*grid_velocity, dealii::EvaluationFlags::values);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  void
  reinit_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(face);
      integrator_grid_velocity_face->gather_evaluate(*grid_velocity,
                                                     dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(face);
      integrator_grid_velocity_face->gather_evaluate(*grid_velocity,
                                                     dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const
  {
    integrator_velocity_m->reinit(cell, face);
    integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(cell, face);
      integrator_grid_velocity_face->gather_evaluate(*grid_velocity,
                                                     dealii::EvaluationFlags::values);
    }

    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      // TODO: Matrix-free implementation in deal.II does currently not allow to access data of
      // the neighboring element in case of cell-based face loops.
      //      integrator_velocity_p->reinit(cell, face);
      //      integrator_velocity_p->gather_evaluate(velocity,dealii::EvaluationFlags::values);
    }
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral. The volume flux depends on the
   * formulation used for the convective term, and is therefore implemented separately for the
   * different formulations (divergence formulation vs. convective formulation). Note that these
   * functions are called by the linearized or linearly implicit convective operator, but not by
   * the nonlinear convective operator.
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_volume_flux_divergence_formulation(vector const & delta_u, unsigned int const q) const
  {
    // u denotes the point of linearization for the linearized problem (of the nonlinear implicit
    // operator) or the transport velocity for the linearly implicit problem
    vector u = get_velocity_cell(q);

    // flux
    tensor F;

    if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
    {
      // linearization of nonlinear convective term

      F = outer_product(u, delta_u);
      F = F + transpose(F);
    }
    else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
    {
      F = outer_product(delta_u, u);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented"));
    }


    // minus sign due to integration by parts
    return -F;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_convective_formulation(vector const &     delta_u,
                                           tensor const &     grad_delta_u,
                                           unsigned int const q) const
  {
    // u denotes the point of linearization for the linearized problem (of the nonlinear implicit
    // operator) or the transport velocity for the linearly implicit problem in case no ALE
    // formulation is used.

    // The velocity w also takes into account the grid velocity u_grid in case of an ALE
    // formulation.

    // w = u
    vector w      = get_velocity_cell(q);
    tensor grad_u = get_velocity_gradient_cell(q);

    // w = u - u_grid
    if(data.ale)
      w -= get_grid_velocity_cell(q);

    // flux
    vector F;

    if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
    {
      // linearization of nonlinear convective term

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      F = grad_u * delta_u + grad_delta_u * w;
    }
    else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
    {
      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      F = grad_delta_u * w;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented"));
    }

    return F;
  }

  /*
   *  Calculates the flux for nonlinear operator on interior faces. This function is needed for
   * face-centric loops and the flux is therefore computed on both sides of an interior face. The
   * interior flux (element m) is the first element in the tuple, the exterior flux (element p,
   * neighbor) is the second element in the tuple.
   */
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, vector>
    calculate_flux_nonlinear_interior_and_neighbor(vector const & uM,
                                                   vector const & uP,
                                                   vector const & normalM,
                                                   vector const & u_grid) const
  {
    vector flux_m, flux_p;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

      flux_m = flux;
      flux_p = -flux; // opposite signs since n⁺ = - n⁻
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector flux;
      scalar average_u_normal = 0.5 * (uM + uP) * normalM;
      if(data.ale)
        average_u_normal -= u_grid * normalM;

      flux = calculate_upwind_flux(uM, uP, average_u_normal);

      // a second term is needed since the strong formulation is implemented (integration by parts
      // twice)
      flux_m = flux - average_u_normal * uM;
      flux_p = -flux + average_u_normal * uP; // opposite signs since n⁺ = - n⁻
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return std::make_tuple(flux_m, flux_p);
  }

  /*
   *  Calculates the flux for nonlinear operator on boundary faces. The flux computation used on
   * interior faces has to be "corrected" if a special outflow boundary condition is used on Neumann
   * boundaries that is able to deal with backflow.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_nonlinear_boundary(vector const &        uM,
                                      vector const &        uP,
                                      vector const &        normalM,
                                      vector const &        u_grid,
                                      BoundaryTypeU const & boundary_type) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      scalar average_u_normal = 0.5 * (uM + uP) * normalM;
      if(data.ale)
        average_u_normal -= u_grid * normalM;

      flux = calculate_upwind_flux(uM, uP, average_u_normal);

      if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
        apply_outflow_bc(flux, average_u_normal);

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_u_normal * uM;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Calculates the flux for linear operator on interior faces. This function is needed for
   * face-centric loops and the flux is therefore computed on both sides of an interior face. The
   * interior flux (element m) is the first element in the tuple, the exterior flux (element p,
   * neighbor) is the second element in the tuple.
   */
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, vector>
    calculate_flux_linear_operator_interior_and_neighbor(vector const &     uM,
                                                         vector const &     uP,
                                                         vector const &     delta_uM,
                                                         vector const &     delta_uP,
                                                         vector const &     normalM,
                                                         unsigned int const q) const
  {
    vector fluxM, fluxP;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term

        fluxM = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
        fluxP = -fluxM;
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        fluxM = calculate_lax_friedrichs_flux_linear_transport(uM, uP, delta_uM, delta_uP, normalM);
        fluxP = -fluxM;
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term

        vector flux = calculate_upwind_flux_linearized(uM, uP, u_grid, delta_uM, delta_uP, normalM);

        scalar average_u_normal = 0.5 * (uM + uP) * normalM;
        if(data.ale)
          average_u_normal -= u_grid * normalM;

        scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fluxM = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
        // opposite signs since n⁺ = - n⁻
        fluxP = -flux + average_u_normal * delta_uP + average_delta_u_normal * uP;
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        // linearly implicit convective term

        vector flux;
        scalar average_u_normal = 0.5 * (uM + uP) * normalM;
        if(data.ale)
          average_u_normal -= u_grid * normalM;

        flux = calculate_upwind_flux(delta_uM, delta_uP, average_u_normal);

        // a second term is needed since the strong formulation is implemented (integration by parts
        // twice)
        fluxM = flux - average_u_normal * delta_uM;
        fluxP = -flux + average_u_normal * delta_uP; // opposite signs since n⁺ = - n⁻
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return std::make_tuple(fluxM, fluxP);
  }

  /*
   *  Calculates the flux for linear operator on interior faces. Only the flux on element e⁻ is
   * computed.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_linear_operator_interior(vector const &     uM,
                                            vector const &     uP,
                                            vector const &     delta_uM,
                                            vector const &     delta_uP,
                                            vector const &     normalM,
                                            unsigned int const q) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term
        flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, delta_uM, delta_uP, normalM);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term

        flux = calculate_upwind_flux_linearized(uM, uP, u_grid, delta_uM, delta_uP, normalM);

        scalar average_u_normal = 0.5 * (uM + uP) * normalM;

        if(data.ale)
          average_u_normal -= u_grid * normalM;

        scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        // linearly implicit convective term

        vector flux;
        scalar average_u_normal = 0.5 * (uM + uP) * normalM;
        if(data.ale)
          average_u_normal -= u_grid * normalM;

        flux = calculate_upwind_flux(delta_uM, delta_uP, average_u_normal);

        // a second term is needed since the strong formulation is implemented (integration by parts
        // twice)
        flux = flux - average_u_normal * delta_uM;
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Calculates the flux for linear operator on boundary faces. The only reason why this
   * function has to be implemented separately is the fact that the flux computation used on
   * interior faces has to be "corrected" if a special outflow boundary condition is used on Neumann
   * boundaries that is able to deal with backflow.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_linear_operator_boundary(vector const &        uM,
                                            vector const &        uP,
                                            vector const &        delta_uM,
                                            vector const &        delta_uP,
                                            vector const &        normalM,
                                            BoundaryTypeU const & boundary_type,
                                            unsigned int const    q) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term
        flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, delta_uM, delta_uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

      if(data.temporal_treatment == TreatmentOfConvectiveTerm::Implicit)
      {
        // linearization of nonlinear convective term
        flux = calculate_upwind_flux_linearized(uM, uP, u_grid, delta_uM, delta_uP, normalM);

        scalar average_u_normal = 0.5 * (uM + uP) * normalM;
        if(data.ale)
          average_u_normal -= u_grid * normalM;

        if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
          apply_outflow_bc(flux, average_u_normal);

        scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
      }
      else if(data.temporal_treatment == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        // linearly implicit convective term

        vector flux;
        scalar average_u_normal = 0.5 * (uM + uP) * normalM;
        if(data.ale)
          average_u_normal -= u_grid * normalM;

        flux = calculate_upwind_flux(delta_uM, delta_uP, average_u_normal);

        if(boundary_type == BoundaryTypeU::Neumann and data.use_outflow_bc == true)
          apply_outflow_bc(flux, average_u_normal);

        // a second term is needed since the strong formulation is implemented (integration by parts
        // twice)
        flux = flux - average_u_normal * delta_uM;
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented"));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Divergence formulation:
   *  Lax-Friedrichs flux
   *  Calculation of lambda according to Shahbazi et al.:
   *  lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
   *         = max ( | 2*(uM)^T*normal | , | 2*(uP)^T*normal | )
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lambda(scalar const & uM_n, scalar const & uP_n) const
  {
    return data.upwind_factor * 2.0 * std::max(std::abs(uM_n), std::abs(uP_n));
  }

  /*
   *  Divergence formulation: Calculate Lax-Friedrichs flux for nonlinear operator.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux(vector const & uM,
                                  vector const & uP,
                                  vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    vector average_normal_flux =
      dealii::make_vectorized_array<Number>(0.5) * (uM * uM_n + uP * uP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Divergence formulation: Calculate Lax-Friedrichs flux for linearly implicit formulation
   * (linear transport with transport velocity w).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linear_transport(vector const & wM,
                                                   vector const & wP,
                                                   vector const & uM,
                                                   vector const & uP,
                                                   vector const & normalM) const
  {
    scalar wM_n = wM * normalM;
    scalar wP_n = wP * normalM;

    vector average_normal_flux =
      dealii::make_vectorized_array<Number>(0.5) * (uM * wM_n + uP * wP_n);

    vector jump_value = uM - uP;

    // the function calculate_lambda() is for the nonlinear operator with quadratic nonlinearity. In
    // case of linear transport, lambda is reduced by a factor of 2.
    scalar lambda = 0.5 * calculate_lambda(wM_n, wP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Divergence formulation:  Calculate Lax-Friedrichs flux for linearized operator.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linearized(vector const & uM,
                                             vector const & uP,
                                             vector const & delta_uM,
                                             vector const & delta_uP,
                                             vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    scalar delta_uM_n = delta_uM * normalM;
    scalar delta_uP_n = delta_uP * normalM;

    vector average_normal_flux =
      dealii::make_vectorized_array<Number>(0.5) *
      (uM * delta_uM_n + delta_uM * uM_n + uP * delta_uP_n + delta_uP * uP_n);

    vector jump_value = delta_uM - delta_uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   * Convective formulation: Calculate upwind flux given an average normal velocity. This function
   * is used for the nonlinear operator and for the linearly implicit treatment of the convective
   * term.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux(vector const & uM,
                          vector const & uP,
                          scalar const & average_normal_velocity) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    vector jump_value = uM - uP;

    return (average_normal_velocity * average_velocity +
            data.upwind_factor * 0.5 * std::abs(average_normal_velocity) * jump_value);
  }

  /*
   *  Convective formulation: Calculate upwind flux for linearized operator.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linearized(vector const & uM,
                                     vector const & uP,
                                     vector const & u_grid,
                                     vector const & delta_uM,
                                     vector const & delta_uP,
                                     vector const & normalM) const
  {
    vector average_velocity       = 0.5 * (uM + uP);
    vector delta_average_velocity = 0.5 * (delta_uM + delta_uP);

    scalar average_normal_velocity = average_velocity * normalM;
    if(data.ale)
      average_normal_velocity -= u_grid * normalM;

    scalar delta_average_normal_velocity = delta_average_velocity * normalM;

    vector jump_value = delta_uM - delta_uP;

    return (average_normal_velocity * delta_average_velocity +
            delta_average_normal_velocity * average_velocity +
            data.upwind_factor * 0.5 * std::abs(average_normal_velocity) * jump_value);
  }

  /*
   * outflow BC according to Gravemeier et al. (2012)
   */
  inline DEAL_II_ALWAYS_INLINE //
    void
    apply_outflow_bc(vector & flux, scalar const & normal_velocity) const
  {
    // we need a factor indicating whether we have inflow or outflow
    // on the Neumann part of the boundary.
    // outflow: factor = 1.0 (do nothing, neutral element of multiplication)
    // inflow:  factor = 0.0 (set convective flux to zero)
    scalar outflow_indicator = dealii::make_vectorized_array<Number>(1.0);

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
    {
      if(normal_velocity[v] < 0.0) // backflow at outflow boundary
        outflow_indicator[v] = 0.0;
    }

    // set flux to zero in case of backflow
    flux = outflow_indicator * flux;
  }

private:
  ConvectiveKernelData data;

  // linearization velocity for nonlinear problems or transport velocity for "linearly implicit
  // formulation" of convective term
  lazy_ptr<VectorType> velocity;

  // grid velocity for ALE problems
  lazy_ptr<VectorType> grid_velocity;

  std::shared_ptr<IntegratorCell> integrator_velocity;
  std::shared_ptr<IntegratorFace> integrator_velocity_m;
  std::shared_ptr<IntegratorFace> integrator_velocity_p;

  std::shared_ptr<IntegratorCell> integrator_grid_velocity;
  std::shared_ptr<IntegratorFace> integrator_grid_velocity_face;
};


} // namespace Operators

template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData() : OperatorBaseData(), quad_index_nonlinear(0)
  {
  }

  /*
   * Needs ConvectiveKernelData since it is not possible to remove all kernel parameters from
   * function do_cell_integral(), i.e. the parameter FormulationConvectiveTerm is explicitly needed
   * by ConvectiveOperator.
   */
  Operators::ConvectiveKernelData kernel_data;

  /*
   * In addition to the quad_index in OperatorBaseData (which denotes the quadrature index for the
   * linearized problem), an additional quadrature index has to be specified for the convective
   * operator evaluation in case of explicit time integration or nonlinear residual evaluation).
   */
  unsigned int quad_index_nonlinear;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};



template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, dim>
{
public:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef ConvectiveOperator<dim, Number> This;

  typedef OperatorBase<dim, Number, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  ConvectiveOperator()
  {
  }

  void
  set_velocity_copy(VectorType const & src) const;

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  initialize(dealii::MatrixFree<dim, Number> const &                   matrix_free,
             dealii::AffineConstraints<Number> const &                 affine_constraints,
             ConvectiveOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel);

  /*
   * Evaluate nonlinear operator.
   */
  void
  evaluate_nonlinear_operator(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_nonlinear_operator_add(VectorType &       dst,
                                  VectorType const & src,
                                  Number const       time) const;

  // these functions are not implemented for the convective operator. They only have to exist
  // due to the definition of the base class.
  void
  evaluate(VectorType & dst, VectorType const & src) const final;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const final;

private:
  /*
   *  Evaluation of nonlinear operator.
   */
  void
  cell_loop_nonlinear_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                            dst,
                               VectorType const &                      src,
                               Range const &                           cell_range) const;

  void
  face_loop_nonlinear_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                            dst,
                               VectorType const &                      src,
                               Range const &                           face_range) const;

  void
  boundary_face_loop_nonlinear_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                        VectorType &                            dst,
                                        VectorType const &                      src,
                                        Range const &                           face_range) const;

  void
  do_cell_integral_nonlinear_operator(IntegratorCell & integrator,
                                      IntegratorCell & integrator_u_grid) const;

  void
  do_face_integral_nonlinear_operator(IntegratorFace & integrator_m,
                                      IntegratorFace & integrator_p,
                                      IntegratorFace & integrator_grid_velocity) const;

  void
  do_boundary_integral_nonlinear_operator(IntegratorFace & integrator,
                                          IntegratorFace & integrator_grid_velocity,
                                          dealii::types::boundary_id const & boundary_id) const;

  /*
   * Linearized operator.
   */

  // Note: this function can only be used for the linearized operator.
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const final;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const final;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const final;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const final;

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const final;

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  ConvectiveOperatorData<dim> operator_data;

  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
