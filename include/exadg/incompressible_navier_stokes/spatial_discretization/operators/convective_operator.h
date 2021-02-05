/*
 * convective_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

namespace Operators
{
struct ConvectiveKernelData
{
  ConvectiveKernelData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror),
      ale(false)
  {
  }

  FormulationConvectiveTerm formulation;

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
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         ConvectiveKernelData const &    data,
         unsigned int const              dof_index,
         unsigned int const              quad_index_linearized,
         bool const                      is_mg)
  {
    this->data = data;

    // integrators for linearized problem
    integrator_velocity.reset(new IntegratorCell(matrix_free, dof_index, quad_index_linearized));
    integrator_velocity_m.reset(
      new IntegratorFace(matrix_free, true, dof_index, quad_index_linearized));
    integrator_velocity_p.reset(
      new IntegratorFace(matrix_free, false, dof_index, quad_index_linearized));

    if(data.ale)
    {
      integrator_grid_velocity.reset(
        new IntegratorCell(matrix_free, dof_index, quad_index_linearized));
      integrator_grid_velocity_face.reset(
        new IntegratorFace(matrix_free, true, dof_index, quad_index_linearized));
    }

    // use own storage of velocity vector only in case of multigrid
    if(is_mg)
    {
      velocity.reset();
      matrix_free.initialize_dof_vector(velocity.own(), dof_index);
    }

    if(data.ale)
    {
      matrix_free.initialize_dof_vector(grid_velocity, dof_index);

      AssertThrow(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation,
                  ExcMessage(
                    "ALE formulation can only be used in combination with ConvectiveFormulation"));
    }
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
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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
  set_velocity_copy(VectorType const & src) const
  {
    velocity.own() = src;

    velocity->update_ghost_values();
  }

  void
  set_velocity_ptr(VectorType const & src) const
  {
    velocity.reset(src);

    velocity->update_ghost_values();
  }

  void
  set_grid_velocity_ptr(VectorType const & src) const
  {
    grid_velocity = src;
    grid_velocity.update_ghost_values();
  }

  VectorType const &
  get_grid_velocity() const
  {
    return grid_velocity;
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
      integrator_velocity->gather_evaluate(*velocity, true, false, false);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      integrator_velocity->gather_evaluate(*velocity, true, true, false);

      if(data.ale)
        integrator_grid_velocity->gather_evaluate(grid_velocity, true, false, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  reinit_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(*velocity, true, false);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(*velocity, true, false);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(face);
      integrator_grid_velocity_face->gather_evaluate(grid_velocity, true, false);
    }
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(*velocity, true, false);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(face);
      integrator_grid_velocity_face->gather_evaluate(grid_velocity, true, false);
    }
  }

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const
  {
    integrator_velocity_m->reinit(cell, face);
    integrator_velocity_m->gather_evaluate(*velocity, true, false);

    if(data.ale)
    {
      integrator_grid_velocity_face->reinit(cell, face);
      integrator_grid_velocity_face->gather_evaluate(grid_velocity, true, false);
    }

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
    get_volume_flux_linearized_divergence_formulation(vector const &     delta_u,
                                                      unsigned int const q) const
  {
    vector u = get_velocity_cell(q);
    tensor F = outer_product(u, delta_u);

    // minus sign due to integration by parts
    return -(F + transpose(F));
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_linearized_convective_formulation(vector const &     delta_u,
                                                      tensor const &     grad_delta_u,
                                                      unsigned int const q) const
  {
    // w = u
    vector w      = get_velocity_cell(q);
    tensor grad_u = get_velocity_gradient_cell(q);

    // w = u - u_grid
    if(data.ale)
      w -= get_grid_velocity_cell(q);

    // plus sign since the strong formulation is used, i.e.
    // integration by parts is performed twice
    vector F = grad_u * delta_u + grad_delta_u * w;

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
                                      vector const &        u_grid,
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
      scalar average_u_normal = 0.5 * (uM + uP) * normalM;
      if(data.ale)
        average_u_normal -= u_grid * normalM;

      flux = calculate_upwind_flux(uM, uP, average_u_normal);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
      {
        if(data.ale == false)
          apply_outflow_bc(flux, uM * normalM);
        else
          apply_outflow_bc(flux, (uM - u_grid) * normalM);
      }

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - average_u_normal * uM;
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
    calculate_flux_linearized_interior_and_neighbor(vector const &     uM,
                                                    vector const &     uP,
                                                    vector const &     delta_uM,
                                                    vector const &     delta_uP,
                                                    vector const &     normalM,
                                                    unsigned int const q) const
  {
    vector fluxM, fluxP;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fluxM = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
      fluxP = -fluxM;
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

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
    calculate_flux_linearized_interior(vector const &     uM,
                                       vector const &     uP,
                                       vector const &     delta_uM,
                                       vector const &     delta_uP,
                                       vector const &     normalM,
                                       unsigned int const q) const
  {
    vector flux;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normalM);
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

      flux = calculate_upwind_flux_linearized(uM, uP, u_grid, delta_uM, delta_uP, normalM);

      scalar average_u_normal = 0.5 * (uM + uP) * normalM;

      if(data.ale)
        average_u_normal -= u_grid * normalM;

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
                                       BoundaryTypeU const & boundary_type,
                                       unsigned int const    q) const
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
      vector u_grid;
      if(data.ale)
        u_grid = get_grid_velocity_face(q);

      flux = calculate_upwind_flux_linearized(uM, uP, u_grid, delta_uM, delta_uP, normalM);

      if(boundary_type == BoundaryTypeU::Neumann && data.use_outflow_bc == true)
        apply_outflow_bc(flux, uM * normalM);

      scalar average_u_normal = 0.5 * (uM + uP) * normalM;
      if(data.ale)
        average_u_normal -= u_grid * normalM;

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
    // outflow: factor = 1.0 (do nothing, neutral element of multiplication)
    // inflow:  factor = 0.0 (set convective flux to zero)
    scalar outflow_indicator = make_vectorized_array<Number>(1.0);

    for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
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

    if(boundary_type == BoundaryTypeU::Dirichlet || boundary_type == BoundaryTypeU::DirichletMortar)
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
  ConvectiveKernelData data;

  mutable lazy_ptr<VectorType> velocity;
  mutable VectorType           grid_velocity;

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

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};



template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, dim>
{
public:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef ConvectiveOperator<dim, Number> This;

  typedef OperatorBase<dim, Number, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  ConvectiveOperator() : velocity_linear_transport(nullptr)
  {
  }

  void
  set_velocity_copy(VectorType const & src) const;

  void
  set_velocity_ptr(VectorType const & src) const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  initialize(MatrixFree<dim, Number> const &                           matrix_free,
             AffineConstraints<Number> const &                         constraint_matrix,
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

  /*
   * Evaluate operator (linear transport with a divergence-free velocity). This function
   * is required in case of operator-integration-factor (OIF) splitting.
   */
  void
  evaluate_linear_transport(VectorType &       dst,
                            VectorType const & src,
                            Number const       time,
                            VectorType const & velocity_linear_transport) const;

  void
  rhs(VectorType & dst) const;

  void
  rhs_add(VectorType & dst) const;

  void
  evaluate(VectorType & dst, VectorType const & src) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const;

private:
  /*
   *  Evaluation of nonlinear operator.
   */
  void
  cell_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   cell_range) const;

  void
  face_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   face_range) const;

  void
  boundary_face_loop_nonlinear_operator(MatrixFree<dim, Number> const & matrix_free,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const;

  void
  do_cell_integral_nonlinear_operator(IntegratorCell & integrator,
                                      IntegratorCell & integrator_u_grid) const;

  void
  do_face_integral_nonlinear_operator(IntegratorFace & integrator_m,
                                      IntegratorFace & integrator_p,
                                      IntegratorFace & integrator_grid_velocity) const;

  void
  do_boundary_integral_nonlinear_operator(IntegratorFace &           integrator,
                                          IntegratorFace &           integrator_grid_velocity,
                                          types::boundary_id const & boundary_id) const;

  /*
   *  OIF splitting: evaluation of convective operator (linear transport).
   */
  IntegratorFlags
  get_integrator_flags_linear_transport() const;

  void
  set_velocity_linear_transport(VectorType const & src) const;

  void
  cell_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   cell_range) const;

  void
  face_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   face_range) const;

  void
  boundary_face_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                                      VectorType &                    dst,
                                      VectorType const &              src,
                                      Range const &                   face_range) const;

  void
  do_cell_integral_linear_transport(IntegratorCell & integrator, IntegratorCell & velocity) const;

  void
  do_face_integral_linear_transport(IntegratorFace & integrator_m,
                                    IntegratorFace & integrator_p,
                                    IntegratorFace & velocity_m,
                                    IntegratorFace & velocity_p) const;

  void
  do_boundary_integral_linear_transport(IntegratorFace &           integrator,
                                        IntegratorFace &           velocity,
                                        types::boundary_id const & boundary_id) const;


  /*
   * Linearized operator.
   */

  // Note: this function can only be used for the linearized operator.
  void
  reinit_cell(unsigned int const cell) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face(unsigned int const face) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_boundary_face(unsigned int const face) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const;

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  ConvectiveOperatorData<dim> operator_data;

  // OIF substepping
  mutable VectorType const * velocity_linear_transport;

  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
