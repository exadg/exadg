/*
 * convective_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../user_interface/input_parameters.h"
#include "weak_boundary_conditions.h"

#include "../../../operators/operator_base.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
struct ConvectiveKernelData
{
  ConvectiveKernelData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror)
  {
  }

  FormulationConvectiveTerm formulation;

  double upwind_factor;

  bool use_outflow_bc;

  TypeDirichletBCs type_dirichlet_bc;
};

template<int dim, typename Number>
class ConvectiveKernel
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         ConvectiveKernelData const &    data_in,
         unsigned int const              dof_index,
         unsigned int const              quad_index) const
  {
    data = data_in;

    matrix_free.initialize_dof_vector(velocity, dof_index);

    integrator_velocity.reset(new IntegratorCell(matrix_free, dof_index, quad_index));
    integrator_velocity_m.reset(new IntegratorFace(matrix_free, true, dof_index, quad_index));
    integrator_velocity_p.reset(new IntegratorFace(matrix_free, false, dof_index, quad_index));
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells          = update_JxW_values | update_gradients;
    flags.inner_faces    = update_JxW_values | update_normal_vectors;
    flags.boundary_faces = update_JxW_values | update_normal_vectors;

    return flags;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flags.cell_evaluate  = CellFlags(true, false, false);
      flags.cell_integrate = CellFlags(false, true, false);

      flags.face_evaluate  = FaceFlags(true, false);
      flags.face_integrate = FaceFlags(true, false);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flags.cell_evaluate  = CellFlags(true, true, false);
      flags.cell_integrate = CellFlags(true, false, false);

      flags.face_evaluate  = FaceFlags(true, false);
      flags.face_integrate = FaceFlags(true, false);
    }
    else if(data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      flags.cell_evaluate  = CellFlags(true, true, false);
      flags.cell_integrate = CellFlags(true, true, false);

      flags.face_evaluate  = FaceFlags(true, false);
      flags.face_integrate = FaceFlags(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flags;
  }

  IntegratorFlags
  get_integrator_flags_linear_transport() const
  {
    IntegratorFlags flags;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flags.cell_evaluate  = CellFlags(true, false, false);
      flags.cell_integrate = CellFlags(false, true, false);

      flags.face_evaluate  = FaceFlags(true, false);
      flags.face_integrate = FaceFlags(true, false);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flags.cell_evaluate  = CellFlags(false, true, false);
      flags.cell_integrate = CellFlags(true, false, false);

      flags.face_evaluate  = FaceFlags(true, false);
      flags.face_integrate = FaceFlags(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flags;
  }

  void
  set_velocity(VectorType const & src) const
  {
    velocity = src;

    velocity.update_ghost_values();
  }

  VectorType const &
  get_velocity() const
  {
    return velocity;
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

  // linear transport
  void
  reinit_cell_linear_transport(unsigned int const cell) const
  {
    // Note that the integrator flags which are valid here are different from those for the
    // linearized operator!
    integrator_velocity->reinit(cell);
    integrator_velocity->gather_evaluate(velocity, true, false, false);
  }

  void
  reinit_face_linear_transport(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(velocity, true, false);
  }

  void
  reinit_boundary_face_linear_transport(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);
  }

  // linearized operator
  void
  reinit_cell(unsigned int const cell) const
  {
    integrator_velocity->reinit(cell);

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
      integrator_velocity->gather_evaluate(velocity, true, false, false);
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
      integrator_velocity->gather_evaluate(velocity, true, true, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  reinit_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(velocity, true, false);
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);
  }

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const
  {
    integrator_velocity_m->reinit(cell, face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      // TODO: Matrix-free implementation in deal.II does currently not allow to access data of
      // the neighboring element in case of cell-based face loops.
      //      integrator_velocity_p->reinit(cell, face);
      //      integrator_velocity_p->gather_evaluate(velocity, true, false);
    }
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral. The volume flux depends on the
   * formulation used for the convective term, and is therefore implemented separately for the
   * different formulations
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_volume_flux_linearized_divergence_formulation(vector const & u,
                                                      vector const & delta_u) const
  {
    tensor F = outer_product(u, delta_u);

    // minus sign due to integration by parts
    return -(F + transpose(F));
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_linearized_convective_formulation(vector const & u,
                                                      vector const & delta_u,
                                                      tensor const & grad_u,
                                                      tensor const & grad_delta_u) const
  {
    // plus sign since the strong formulation is used, i.e.
    // integration by parts is performed twice
    vector F = grad_u * delta_u + grad_delta_u * u;

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
                                                   vector const & normalM) const
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
      vector flux = calculate_upwind_flux(uM, uP, normalM);

      // a second term is needed since the strong formulation is implemented (integration by parts
      // twice)
      scalar average_u_normal = 0.5 * (uM + uP) * normalM;
      flux_m                  = flux - average_u_normal * uM;
      flux_p                  = -flux + average_u_normal * uP; // opposite signs since n⁺ = - n⁻
    }
    else if(data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

      // corrections to obtain an energy preserving flux (which is not conservative!)
      vector jump = uM - uP;
      flux_m      = flux + 0.25 * jump * normalM * uP;
      flux_p      = -flux + 0.25 * jump * normalM * uM; // opposite signs since n⁺ = - n⁻
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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
                                      BoundaryTypeU const & boundary_type) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flux = calculate_upwind_flux(uM, uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);

      scalar average_u_normal = 0.5 * (uM + uP) * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_u_normal * uM;
    }
    else if(data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);

      // corrections to obtain an energy preserving flux (which is not conservative!)
      flux = flux + 0.25 * (uM - uP) * normalM * uP;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Calculates the flux for operator with linear transport velocity w on interior faces. This
   * function is needed for face-centric loops and the flux is therefore computed on both sides of
   * an interior face. The interior flux (element m) is the first element in the tuple, the exterior
   * flux (element p, neighbor) is the second element in the tuple.
   */
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, vector>
    calculate_flux_linear_transport_interior_and_neighbor(vector const & uM,
                                                          vector const & uP,
                                                          vector const & wM,
                                                          vector const & wP,
                                                          vector const & normalM) const
  {
    vector fluxM, fluxP;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      vector flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wP, normalM);

      fluxM = flux;
      fluxP = -flux; // minus sign since n⁺ = - n⁻
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector flux = calculate_upwind_flux_linear_transport(uM, uP, wM, wP, normalM);

      scalar average_w_normal = 0.5 * (wM + wP) * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      fluxM = flux - average_w_normal * uM;
      fluxP = -flux + average_w_normal * uP; // opposite signs since n⁺ = - n⁻
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return std::make_tuple(fluxM, fluxP);
  }

  /*
   *  Calculates the flux for operator with linear transport velocity w on boundary faces. The flux
   * computation used on interior faces has to be "corrected" if a special outflow boundary
   * condition is used on Neumann boundaries that is able to deal with backflow.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_linear_transport_boundary(vector const &        uM,
                                             vector const &        uP,
                                             vector const &        wM,
                                             vector const &        wP,
                                             vector const &        normalM,
                                             BoundaryTypeU const & boundary_type) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, wM * normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flux = calculate_upwind_flux_linear_transport(uM, uP, wM, wP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, wM * normalM);

      scalar average_w_normal = wM * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_w_normal * uM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Calculates the flux for linearized operator on interior faces. This function is needed for
   * face-centric loops and the flux is therefore computed on both sides of an interior face. The
   * interior flux (element m) is the first element in the tuple, the exterior flux (element p,
   * neighbor) is the second element in the tuple.
   */
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, vector>
    calculate_flux_linearized_interior_and_neighbor(vector const & uM,
                                                    vector const & uP,
                                                    vector const & delta_uM,
                                                    vector const & delta_uP,
                                                    vector const & normalM) const
  {
    vector fluxM, fluxP;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fluxM = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
      fluxP = -fluxM;
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector flux = calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);

      scalar average_u_normal       = 0.5 * (uM + uP) * normalM;
      scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      fluxM = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
      // opposite signs since n⁺ = - n⁻
      fluxP = -flux + average_u_normal * delta_uP + average_delta_u_normal * uP;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return std::make_tuple(fluxM, fluxP);
  }

  /*
   *  Calculates the flux for linearized operator on interior faces. Only the flux on element e⁻ is
   * computed.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_linearized_interior(vector const & uM,
                                       vector const & uP,
                                       vector const & delta_uM,
                                       vector const & delta_uP,
                                       vector const & normalM) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flux = calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);

      scalar average_u_normal       = 0.5 * (uM + uP) * normalM;
      scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Calculates the flux for linearized operator on boundary faces. The only reason why this
   * function has to be implemented separately is the fact that the flux computation used on
   * interior faces has to be "corrected" if a special outflow boundary condition is used on Neumann
   * boundaries that is able to deal with backflow.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux_linearized_boundary(vector const &        uM,
                                       vector const &        uP,
                                       vector const &        delta_uM,
                                       vector const &        delta_uP,
                                       vector const &        normalM,
                                       BoundaryTypeU const & boundary_type) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flux = calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);

      scalar average_u_normal       = 0.5 * (uM + uP) * normalM;
      scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return flux;
  }

  /*
   *  Lax-Friedrichs flux (divergence formulation)
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
   *  Calculate Lax-Friedrichs flux for nonlinear operator (divergence formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux(vector const & uM,
                                  vector const & uP,
                                  vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    vector average_normal_flux = make_vectorized_array<Number>(0.5) * (uM * uM_n + uP * uP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate Lax-Friedrichs flux for nonlinear operator (linear transport).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linear_transport(vector const & uM,
                                                   vector const & uP,
                                                   vector const & wM,
                                                   vector const & wP,
                                                   vector const & normalM) const
  {
    scalar wM_n = wM * normalM;
    scalar wP_n = wP * normalM;

    vector average_normal_flux = make_vectorized_array<Number>(0.5) * (uM * wM_n + uP * wP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(wM_n, wP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator (divergence formulation).
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
      make_vectorized_array<Number>(0.5) *
      (uM * delta_uM_n + delta_uM * uM_n + uP * delta_uP_n + delta_uP * uP_n);

    vector jump_value = delta_uM - delta_uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate upwind flux for nonlinear operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux(vector const & uM, vector const & uP, vector const & normalM) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    scalar average_normal_velocity = average_velocity * normalM;

    vector jump_value = uM - uP;

    return (average_normal_velocity * average_velocity +
            data.upwind_factor * 0.5 * std::abs(average_normal_velocity) * jump_value);
  }

  /*
   *  Calculate upwind flux for convective operator (linear transport, OIF splitting).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linear_transport(vector const & uM,
                                           vector const & uP,
                                           vector const & wM,
                                           vector const & wP,
                                           vector const & normalM) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    vector jump_value = uM - uP;

    scalar average_normal_velocity = 0.5 * (wM + wP) * normalM;

    return (average_normal_velocity * average_velocity +
            data.upwind_factor * 0.5 * std::abs(average_normal_velocity) * jump_value);
  }

  /*
   * outflow BC according to Gravemeier et al. (2012)
   */
  inline DEAL_II_ALWAYS_INLINE //
    void
    apply_outflow_bc(vector & flux, scalar const & uM_n) const
  {
    // we need a factor indicating whether we have inflow or outflow
    // on the Neumann part of the boundary.
    // outflow: factor =  1.0 (do nothing, neutral element of multiplication)
    // inflow:  factor = 0.0 (set convective flux to zero)
    scalar outflow_indicator = make_vectorized_array<Number>(1.0);

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(uM_n[v] < 0.0) // backflow at outflow boundary
        outflow_indicator[v] = 0.0;
    }

    // set flux to zero in case of backflow
    flux = outflow_indicator * flux;
  }

  /*
   *  Calculate upwind flux for linearized operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linearized(vector const & uM,
                                     vector const & uP,
                                     vector const & delta_uM,
                                     vector const & delta_uP,
                                     vector const & normalM) const
  {
    vector average_velocity       = 0.5 * (uM + uP);
    vector delta_average_velocity = 0.5 * (delta_uM + delta_uP);

    scalar average_normal_velocity       = average_velocity * normalM;
    scalar delta_average_normal_velocity = delta_average_velocity * normalM;

    vector jump_value = delta_uM - delta_uP;

    return (average_normal_velocity * delta_average_velocity +
            delta_average_normal_velocity * average_velocity +
            data.upwind_factor * 0.5 * std::abs(average_normal_velocity) * jump_value);
  }

  /*
   * Velocity:
   *
   *  This function calculates the exterior velocity on boundary faces
   *  according to:
   *
   *  Dirichlet boundary: u⁺ = -u⁻ + 2g
   *  Neumann boundary:   u⁺ = u⁻
   *  symmetry boundary:  u⁺ = u⁻ -(u⁻*n)n - (u⁻*n)n = u⁻ - 2 (u⁻*n)n
   *
   *  The name "nonlinear" indicates that this function is used when
   *  evaluating the nonlinear convective operator.
   */
  inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, VectorizedArray<Number>>
      calculate_exterior_value_nonlinear(
        Tensor<1, dim, VectorizedArray<Number>> const & uM,
        unsigned int const                              q,
        FaceIntegrator<dim, dim, Number> &              integrator,
        BoundaryTypeU const &                           boundary_type,
        types::boundary_id const                        boundary_id,
        std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor,
        double const &                                  time) const
  {
    Tensor<1, dim, VectorizedArray<Number>> uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it = boundary_descriptor->dirichlet_bc.find(boundary_id);
      Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

      Tensor<1, dim, VectorizedArray<Number>> g =
        evaluate_vectorial_function(it->second, q_points, time);

      if(data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        uP = -uM + make_vectorized_array<Number>(2.0) * g;
      }
      else if(data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        uP = g;
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      uP = uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      Tensor<1, dim, VectorizedArray<Number>> normalM = integrator.get_normal_vector(q);

      uP = uM - 2. * (uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return uP;
  }

  /*
   * Velocity:
   *
   *  Linearized convective operator (= homogeneous operator):
   *  Dirichlet boundary: delta_u⁺ = - delta_u⁻ or 0
   *  Neumann boundary:   delta_u⁺ = + delta_u⁻
   *  symmetry boundary:  delta_u⁺ = delta_u⁻ - 2 (delta_u⁻*n)n
   */
  inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, VectorizedArray<Number>>
      calculate_exterior_value_linearized(Tensor<1, dim, VectorizedArray<Number>> & delta_uM,
                                          unsigned int const                        q,
                                          FaceIntegrator<dim, dim, Number> &        integrator,
                                          BoundaryTypeU const & boundary_type) const
  {
    // element e⁺
    Tensor<1, dim, VectorizedArray<Number>> delta_uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        delta_uP = -delta_uM;
      }
      else if(data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        // delta_uP = 0
        // do nothing, delta_uP is already initialized with zero
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      delta_uP = delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      Tensor<1, dim, VectorizedArray<Number>> normalM = integrator.get_normal_vector(q);
      delta_uP = delta_uM - 2. * (delta_uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return delta_uP;
  }

private:
  mutable ConvectiveKernelData data;

  mutable VectorType velocity;

  mutable std::shared_ptr<IntegratorCell> integrator_velocity;
  mutable std::shared_ptr<IntegratorFace> integrator_velocity_m;
  mutable std::shared_ptr<IntegratorFace> integrator_velocity_p;
};


} // namespace Operators

template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::ConvectiveKernelData kernel_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};



template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, ConvectiveOperatorData<dim>, dim>
{
public:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef ConvectiveOperator<dim, Number> This;

  typedef OperatorBase<dim, Number, ConvectiveOperatorData<dim>, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  void
  set_solution_linearization(VectorType const & src) const
  {
    kernel.set_velocity(src);
  }

  VectorType const &
  get_solution_linearization() const
  {
    return kernel.get_velocity();
  }

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         AffineConstraints<double> const &   constraint_matrix,
         ConvectiveOperatorData<dim> const & operator_data) const
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    kernel.reinit(matrix_free,
                  operator_data.kernel_data,
                  operator_data.dof_index,
                  operator_data.quad_index);

    this->integrator_flags = kernel.get_integrator_flags();
  }


  /*
   * Evaluate nonlinear operator.
   */
  void
  evaluate_nonlinear_operator(VectorType &       dst,
                              VectorType const & src,
                              Number const       evaluation_time) const
  {
    this->eval_time = evaluation_time;

    this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                            &This::face_loop_nonlinear_operator,
                            &This::boundary_face_loop_nonlinear_operator,
                            this,
                            dst,
                            src,
                            true /*zero_dst_vector = true*/,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  evaluate_nonlinear_operator_add(VectorType &       dst,
                                  VectorType const & src,
                                  Number const       evaluation_time) const
  {
    this->eval_time = evaluation_time;

    this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                            &This::face_loop_nonlinear_operator,
                            &This::boundary_face_loop_nonlinear_operator,
                            this,
                            dst,
                            src,
                            false /*zero_dst_vector = false*/,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Evaluate operator (linear transport with a divergence-free velocity). This function
   * is required in case of operator-integration-factor (OIF) splitting.
   */
  void
  evaluate_linear_transport(VectorType &       dst,
                            VectorType const & src,
                            Number const       evaluation_time,
                            VectorType const & velocity_transport) const
  {
    set_solution_linearization(velocity_transport);

    this->eval_time = evaluation_time;

    this->matrix_free->loop(&This::cell_loop_linear_transport,
                            &This::face_loop_linear_transport,
                            &This::boundary_face_loop_linear_transport,
                            this,
                            dst,
                            src,
                            true /*zero_dst_vector = true*/,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  rhs(VectorType & dst) const
  {
    (void)dst;

    AssertThrow(false,
                ExcMessage("The function rhs() does not make sense for the convective operator."));
  }

  void
  rhs_add(VectorType & dst) const
  {
    (void)dst;

    AssertThrow(
      false, ExcMessage("The function rhs_add() does not make sense for the convective operator."));
  }

  void
  evaluate(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;

    AssertThrow(false,
                ExcMessage(
                  "The function evaluate() does not make sense for the convective operator."));
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;

    AssertThrow(false,
                ExcMessage(
                  "The function evaluate_add() does not make sense for the convective operator."));
  }

private:
  /*
   *  Evaluation of nonlinear operator.
   */
  void
  cell_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   cell_range) const
  {
    (void)matrix_free;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      this->integrator->reinit(cell);

      this->integrator->gather_evaluate(src,
                                        this->integrator_flags.cell_evaluate.value,
                                        this->integrator_flags.cell_evaluate.gradient,
                                        this->integrator_flags.cell_evaluate.hessian);

      do_cell_integral_nonlinear_operator(*this->integrator);

      this->integrator->integrate_scatter(this->integrator_flags.cell_integrate.value,
                                          this->integrator_flags.cell_integrate.gradient,
                                          dst);
    }
  }

  void
  face_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   face_range) const
  {
    (void)matrix_free;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);
      this->integrator_p->reinit(face);

      this->integrator_m->gather_evaluate(src,
                                          this->integrator_flags.face_evaluate.value,
                                          this->integrator_flags.face_evaluate.gradient);

      this->integrator_p->gather_evaluate(src,
                                          this->integrator_flags.face_evaluate.value,
                                          this->integrator_flags.face_evaluate.gradient);

      do_face_integral_nonlinear_operator(*this->integrator_m, *this->integrator_p);

      this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                            this->integrator_flags.face_integrate.gradient,
                                            dst);

      this->integrator_p->integrate_scatter(this->integrator_flags.face_integrate.value,
                                            this->integrator_flags.face_integrate.gradient,
                                            dst);
    }
  }

  void
  boundary_face_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const
  {
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);

      this->integrator_m->gather_evaluate(src,
                                          this->integrator_flags.face_evaluate.value,
                                          this->integrator_flags.face_evaluate.gradient);

      do_boundary_integral_nonlinear_operator(*this->integrator_m,
                                              matrix_free.get_boundary_id(face));

      this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                            this->integrator_flags.face_integrate.gradient,
                                            dst);
    }
  }

  void
  do_cell_integral_nonlinear_operator(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector u = integrator.get_value(q);

      if(this->operator_data.kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        // nonlinear convective flux F(u) = uu
        tensor F = outer_product(u, u);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
      else if(this->operator_data.kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        // convective formulation: (u * grad) u = grad(u) * u
        tensor gradient_u = integrator.get_gradient(q);
        vector F          = gradient_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
      else if(this->operator_data.kernel_data.formulation ==
              FormulationConvectiveTerm::EnergyPreservingFormulation)
      {
        // nonlinear convective flux F(u) = uu
        tensor F          = outer_product(u, u);
        scalar divergence = integrator.get_divergence(q);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
        integrator.submit_value(-0.5 * divergence * u, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }
  }

  void
  do_face_integral_nonlinear_operator(IntegratorFace & integrator_m,
                                      IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m      = integrator_m.get_value(q);
      vector u_p      = integrator_p.get_value(q);
      vector normal_m = integrator_m.get_normal_vector(q);

      std::tuple<vector, vector> flux =
        kernel.calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m);

      integrator_m.submit_value(std::get<0>(flux), q);
      integrator_p.submit_value(std::get<1>(flux), q);
    }
  }

  void
  do_boundary_integral_nonlinear_operator(IntegratorFace &           integrator,
                                          types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector u_m = integrator.get_value(q);
      vector u_p = kernel.calculate_exterior_value_nonlinear(
        u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

      vector normal_m = integrator.get_normal_vector(q);

      vector flux = kernel.calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, boundary_type);

      integrator.submit_value(flux, q);
    }
  }

  /*
   *  OIF splitting: evaluation of convective operator (linear transport).
   */
  void
  cell_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   cell_range) const
  {
    (void)matrix_free;

    IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      this->integrator->reinit(cell);

      this->integrator->gather_evaluate(src,
                                        flags.cell_evaluate.value,
                                        flags.cell_evaluate.gradient,
                                        flags.cell_evaluate.hessian);

      kernel.reinit_cell_linear_transport(cell);

      do_cell_integral_linear_transport(*this->integrator);

      this->integrator->integrate_scatter(flags.cell_integrate.value,
                                          flags.cell_integrate.gradient,
                                          dst);
    }
  }

  void
  face_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   face_range) const
  {
    (void)matrix_free;

    IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);
      this->integrator_p->reinit(face);

      this->integrator_m->gather_evaluate(src,
                                          flags.face_evaluate.value,
                                          flags.face_evaluate.gradient);

      this->integrator_p->gather_evaluate(src,
                                          flags.face_evaluate.value,
                                          flags.face_evaluate.gradient);

      kernel.reinit_face_linear_transport(face);

      do_face_integral_linear_transport(*this->integrator_m, *this->integrator_p);

      this->integrator_m->integrate_scatter(flags.face_integrate.value,
                                            flags.face_integrate.gradient,
                                            dst);

      this->integrator_p->integrate_scatter(flags.face_integrate.value,
                                            flags.face_integrate.gradient,
                                            dst);
    }
  }

  void
  boundary_face_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                                      VectorType &                    dst,
                                      VectorType const &              src,
                                      Range const &                   face_range) const
  {
    IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);
      this->integrator_m->gather_evaluate(src,
                                          flags.face_evaluate.value,
                                          flags.face_evaluate.gradient);

      kernel.reinit_boundary_face_linear_transport(face);

      do_boundary_integral_linear_transport(*this->integrator_m, matrix_free.get_boundary_id(face));

      this->integrator_m->integrate_scatter(flags.face_integrate.value,
                                            flags.face_integrate.gradient,
                                            dst);
    }
  }

  void
  do_cell_integral_linear_transport(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector w = kernel.get_velocity_cell(q);

      if(this->operator_data.kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        // nonlinear convective flux F = uw
        vector u = integrator.get_value(q);
        tensor F = outer_product(u, w);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
      else if(this->operator_data.kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        // convective formulation: grad(u) * w
        tensor grad_u = integrator.get_gradient(q);
        vector F      = grad_u * w;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }
  }

  void
  do_face_integral_linear_transport(IntegratorFace & integrator_m,
                                    IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m = integrator_m.get_value(q);
      vector u_p = integrator_p.get_value(q);

      vector w_m = kernel.get_velocity_m(q);
      vector w_p = kernel.get_velocity_p(q);

      vector normal_m = integrator_m.get_normal_vector(q);

      std::tuple<vector, vector> flux =
        kernel.calculate_flux_linear_transport_interior_and_neighbor(u_m, u_p, w_m, w_p, normal_m);

      integrator_m.submit_value(std::get<0>(flux), q);
      integrator_p.submit_value(std::get<1>(flux), q);
    }
  }

  void
  do_boundary_integral_linear_transport(IntegratorFace &           integrator,
                                        types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector u_m = integrator.get_value(q);
      vector u_p = kernel.calculate_exterior_value_nonlinear(
        u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

      // concerning the transport velocity w, use the same value for interior and
      // exterior states, i.e., do not prescribe boundary conditions
      vector w_m = kernel.get_velocity_m(q);

      vector normal_m = integrator.get_normal_vector(q);

      vector flux = kernel.calculate_flux_linear_transport_boundary(
        u_m, u_p, w_m, w_m, normal_m, boundary_type);

      integrator.submit_value(flux, q);
    }
  }


  /*
   * Linearized operator.
   */

  // Note: this function can only be used for the linearized operator.
  void
  reinit_cell(unsigned int const cell) const
  {
    Base::reinit_cell(cell);

    kernel.reinit_cell(cell);
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face(unsigned int const face) const
  {
    Base::reinit_face(face);

    kernel.reinit_face(face);
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_boundary_face(unsigned int const face) const
  {
    Base::reinit_boundary_face(face);

    kernel.reinit_boundary_face(face);
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const
  {
    Base::reinit_face_cell_based(cell, face, boundary_id);

    kernel.reinit_face_cell_based(cell, face, boundary_id);
  }

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector delta_u = integrator.get_value(q);
      vector u       = kernel.get_velocity_cell(q);

      if(this->operator_data.kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        tensor flux = kernel.get_volume_flux_linearized_divergence_formulation(u, delta_u);

        integrator.submit_gradient(flux, q);
      }
      else if(this->operator_data.kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        tensor grad_u       = kernel.get_velocity_gradient_cell(q);
        tensor grad_delta_u = integrator.get_gradient(q);

        vector flux = kernel.get_volume_flux_linearized_convective_formulation(u,
                                                                               delta_u,
                                                                               grad_u,
                                                                               grad_delta_u);

        integrator.submit_value(flux, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }
  }

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m = kernel.get_velocity_m(q);
      vector u_p = kernel.get_velocity_p(q);

      vector delta_u_m = integrator_m.get_value(q);
      vector delta_u_p = integrator_p.get_value(q);

      vector normal_m = integrator_m.get_normal_vector(q);

      std::tuple<vector, vector> flux = kernel.calculate_flux_linearized_interior_and_neighbor(
        u_m, u_p, delta_u_m, delta_u_p, normal_m);

      integrator_m.submit_value(std::get<0>(flux) /* flux_m */, q);
      integrator_p.submit_value(std::get<1>(flux) /* flux_p */, q);
    }
  }

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m = kernel.get_velocity_m(q);
      vector u_p = kernel.get_velocity_p(q);

      vector delta_u_m = integrator_m.get_value(q);
      vector delta_u_p; // set exterior value to zero

      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux =
        kernel.calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

      integrator_m.submit_value(flux, q);
    }
  }

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m = kernel.get_velocity_m(q);
      // TODO
      // Accessing exterior data is currently not available in deal.II/matrixfree.
      // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
      // are not calculated exactly.
      vector u_p = u_m;

      vector delta_u_m = integrator_m.get_value(q);
      vector delta_u_p; // set exterior value to zero

      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux =
        kernel.calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

      integrator_m.submit_value(flux, q);
    }
  }

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_m;

    for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
    {
      vector u_m = kernel.get_velocity_m(q);
      vector u_p = kernel.get_velocity_p(q);

      vector delta_u_m; // set exterior value to zero
      vector delta_u_p = integrator_p.get_value(q);

      vector normal_p = -integrator_p.get_normal_vector(q);

      vector flux =
        kernel.calculate_flux_linearized_interior(u_p, u_m, delta_u_p, delta_u_m, normal_p);

      integrator_p.submit_value(flux, q);
    }
  }

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    // make sure that this function is only accessed for OperatorType::homogeneous
    AssertThrow(
      operator_type == OperatorType::homogeneous,
      ExcMessage(
        "For the linearized convective operator, only OperatorType::homogeneous makes sense."));

    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector u_m = kernel.get_velocity_m(q);
      vector u_p = kernel.calculate_exterior_value_nonlinear(
        u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

      vector delta_u_m = integrator.get_value(q);
      vector delta_u_p =
        kernel.calculate_exterior_value_linearized(delta_u_m, q, integrator, boundary_type);

      vector normal_m = integrator.get_normal_vector(q);

      vector flux = kernel.calculate_flux_linearized_boundary(
        u_m, u_p, delta_u_m, delta_u_p, normal_m, boundary_type);

      integrator.submit_value(flux, q);
    }
  }

  Operators::ConvectiveKernel<dim, Number> kernel;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
