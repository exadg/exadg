/*
 * convective_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

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
      linearization_type(LinearizationType::Newton),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror),
      ale(false)
  {
  }

  FormulationConvectiveTerm formulation;

  LinearizationType linearization_type;

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
         bool const                              is_mg)
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

    // use own storage of velocity vector only in case of multigrid
    if(is_mg)
    {
      velocity.reset();
      matrix_free.initialize_dof_vector(velocity.own(), dof_index);
    }

    if(data.ale)
    {
      matrix_free.initialize_dof_vector(grid_velocity.own(), dof_index);

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
  set_velocity_ptr(VectorType const & src)
  {
    velocity.reset(src);

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
   * different formulations
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_volume_flux_linearized_divergence_formulation(vector const &     delta_u,
                                                      unsigned int const q) const
  {
    vector u = get_velocity_cell(q);

    tensor F;

    // minus sign due to integration by parts
    if(data.linearization_type == LinearizationType::Newton)
    {
      F = outer_product(u, delta_u);
      F = -(F + transpose(F));
  	}
    else if(data.linearization_type == LinearizationType::Picard)
    {
      F = -outer_product(u, delta_u);
  	}
    else
    {
      AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
    }

    return F;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_linearized_convective_formulation(vector const &     delta_u,
                                                      tensor const &     grad_delta_u,
                                                      unsigned int const q) const
  {
    // w = u
    vector w = get_velocity_cell(q);

    // w = u - u_grid
    if(data.ale)
      w -= get_grid_velocity_cell(q);

    vector F;

    // plus sign since the strong formulation is used, i.e.
    // integration by parts is performed twice
    if(data.linearization_type == LinearizationType::Newton)
    {
      tensor grad_u = get_velocity_gradient_cell(q);
      F = grad_u * delta_u + grad_delta_u * w;
    }
    else if(data.linearization_type == LinearizationType::Picard)
    {
      F = grad_delta_u * w;
    }
    else
    {
  	  AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
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
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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

      if(data.linearization_type == LinearizationType::Newton)
      {
        scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fluxM = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
        // opposite signs since n⁺ = - n⁻
        fluxP = -flux + average_u_normal * delta_uP + average_delta_u_normal * uP;
      }
      else if(data.linearization_type == LinearizationType::Picard)
      {
        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fluxM = flux - average_u_normal * delta_uM;
        // opposite signs since n⁺ = - n⁻
        fluxP = -flux + average_u_normal * delta_uP;
      }
      else
      {
    	AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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

      if(data.linearization_type == LinearizationType::Newton)
      {
		scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

		// second term appears since the strong formulation is implemented (integration by parts
		// is performed twice)
		flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
      }
      else if(data.linearization_type == LinearizationType::Picard)
      {
        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        flux = flux - average_u_normal * delta_uM;
      }
      else
      {
    	AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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

      if(data.linearization_type == LinearizationType::Newton)
      {
        scalar average_delta_u_normal = 0.5 * (delta_uM + delta_uP) * normalM;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        flux = flux - average_u_normal * delta_uM - average_delta_u_normal * uM;
      }
      else if(data.linearization_type == LinearizationType::Picard)
      {
        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        flux = flux - average_u_normal * delta_uM;
      }
      else
      {
    	AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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

    vector average_normal_flux =
      dealii::make_vectorized_array<Number>(0.5) * (uM * uM_n + uP * uP_n);

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

    vector average_normal_flux =
      dealii::make_vectorized_array<Number>(0.5) * (uM * wM_n + uP * wP_n);

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

    vector jump_value = delta_uM - delta_uP;
    scalar lambda = calculate_lambda(uM_n, uP_n);

    vector average_normal_flux;
    if(data.linearization_type == LinearizationType::Newton)
    {
      scalar delta_uM_n = delta_uM * normalM;
      scalar delta_uP_n = delta_uP * normalM;

      average_normal_flux =
        dealii::make_vectorized_array<Number>(0.5) *
        (uM * delta_uM_n + delta_uM * uM_n + uP * delta_uP_n + delta_uP * uP_n);
    }
    else if(data.linearization_type == LinearizationType::Picard)
    {
      average_normal_flux =
    	dealii::make_vectorized_array<Number>(0.5) * (delta_uM * uM_n + delta_uP * uP_n);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
    }

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
    scalar outflow_indicator = dealii::make_vectorized_array<Number>(1.0);

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
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

    vector jump_value = delta_uM - delta_uP;

    vector flux;
    if (data.linearization_type == LinearizationType::Newton)
    {
      scalar delta_average_normal_velocity = delta_average_velocity * normalM;

	  flux = average_normal_velocity * delta_average_velocity +
	    delta_average_normal_velocity * average_velocity +
	    (data.upwind_factor * 0.5 * std::abs(average_normal_velocity)) * jump_value;
    }
    else if(data.linearization_type == LinearizationType::Picard)
    {
      flux = average_normal_velocity * delta_average_velocity +
        (data.upwind_factor * 0.5 * std::abs(average_normal_velocity)) * jump_value;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
    }

    return flux;
  }

  /*
   * Velocity:
   *
   *  Linearized convective operator
   *  homogeneous operator for Newton linearization (g = 0)
   *  inhomogeneous operator for Picard linearization
   *
   *  Dirichlet boundary: delta_u⁺ = - delta_u⁻ + 2 g or g
   *  Neumann boundary:   delta_u⁺ = + delta_u⁻
   *  symmetry boundary:  delta_u⁺ = delta_u⁻ - 2 (delta_u⁻*n)n
   *
   */
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    calculate_exterior_value_linearized(
      dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> & delta_uM,
      unsigned int const                                        q,
      FaceIntegrator<dim, dim, Number> &                        integrator,
      OperatorType const &                                      operator_type,
      BoundaryTypeU const &                                     boundary_type,
      dealii::types::boundary_id const                          boundary_id,
      std::shared_ptr<BoundaryDescriptorU<dim> const>           boundary_descriptor,
      double const &                                            time) const
  {
    // element e⁺
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> delta_uP;

    if(boundary_type == BoundaryTypeU::Dirichlet || boundary_type == BoundaryTypeU::DirichletCached)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> g;

        if(data.linearization_type == LinearizationType::Newton)
        {
          // g is initialized with 0
        }
        else if(data.linearization_type == LinearizationType::Picard)
        {
          if(boundary_type == BoundaryTypeU::Dirichlet)
          {
            auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
            auto q_points = integrator.quadrature_point(q);

            g = FunctionEvaluator<1, dim, Number>::value(bc, q_points, time);
          }
          else if(boundary_type == BoundaryTypeU::DirichletCached)
          {
            auto bc = boundary_descriptor->dirichlet_cached_bc.find(boundary_id)->second;
            g       = FunctionEvaluator<1, dim, Number>::value(bc,
                                                       integrator.get_current_cell_index(),
                                                       q,
                                                       integrator.get_quadrature_index());
          }
          else
          {
            AssertThrow(false, dealii::ExcMessage("Not implemented."));
          }
        }
        else
        {
	      AssertThrow(false, dealii::ExcMessage("Linearization type not implemented."));
        }

        if(data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
        {
          delta_uP = -delta_uM + 2.0 * g; // dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>(2.0 * g);
        }
        else if(data.type_dirichlet_bc == TypeDirichletBCs::Direct)
        {
          delta_uP = g; // dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>(g);
        }
        else
        {
          AssertThrow(
            false,
            dealii::ExcMessage(
              "Type of imposition of Dirichlet BC's for convective term is not implemented."));
        }
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        delta_uP = -delta_uM;
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      delta_uP = delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> normalM =
        integrator.get_normal_vector(q);
      delta_uP = delta_uM - 2. * (delta_uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
    }


    return delta_uP;
  }


private:
  ConvectiveKernelData data;

  lazy_ptr<VectorType> velocity;
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

  void
  set_velocity_ptr(VectorType const & src) const;

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

  // evaluate the right hand side to be used within Picard linearization
  void
  rhs(VectorType & dst) const;

  void
  rhs_add(VectorType & dst) const;

  // these functions are not implemented for the convective operator. They only have to exist
  // due to the definition of the base class.
  void
  evaluate(VectorType & dst, VectorType const & src) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const;

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
  reinit_cell(unsigned int const cell) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face(unsigned int const face) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_boundary_face(unsigned int const face) const;

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const;

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
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const;

  void
  boundary_face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           face_range) const;

  ConvectiveOperatorData<dim> operator_data;

  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
