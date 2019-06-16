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
    : formulation(FormulationConvectiveTerm::DivergenceFormulation), upwind_factor(1.0)
  {
  }

  FormulationConvectiveTerm formulation;

  double upwind_factor;
};

template<int dim, typename Number>
class ConvectiveKernel
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  void
  reinit(ConvectiveKernelData const & data_in) const
  {
    data = data_in;
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

private:
  mutable ConvectiveKernelData data;
};


} // namespace Operators

template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror)
  {
  }

  bool use_outflow_bc;

  TypeDirichletBCs type_dirichlet_bc;

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
    velocity = src;

    velocity.update_ghost_values();
  }

  VectorType const &
  get_solution_linearization() const
  {
    return velocity;
  }

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         AffineConstraints<double> const &   constraint_matrix,
         ConvectiveOperatorData<dim> const & operator_data) const
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    kernel.reinit(operator_data.kernel_data);

    // TODO -> shift to kernel
    this->matrix_free->initialize_dof_vector(velocity, operator_data.dof_index);

    integrator_velocity.reset(new IntegratorCell(matrix_free,
                                                 this->operator_data.dof_index,
                                                 this->operator_data.quad_index));
    integrator_velocity_m.reset(new IntegratorFace(
      matrix_free, true, this->operator_data.dof_index, this->operator_data.quad_index));
    integrator_velocity_p.reset(new IntegratorFace(
      matrix_free, false, this->operator_data.dof_index, this->operator_data.quad_index));
    // TODO -> shift to kernel

    this->integrator_flags = kernel.get_integrator_flags();

    // OIF splitting
    integrator_flags_linear_transport = kernel.get_integrator_flags_linear_transport();
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
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u = integrator.get_value(q);
        tensor F = outer_product(u, u);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective formulation: (u * grad) u = grad(u) * u
        vector u          = integrator.get_value(q);
        tensor gradient_u = integrator.get_gradient(q);
        vector F          = gradient_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u          = integrator.get_value(q);
        tensor F          = outer_product(u, u);
        scalar divergence = integrator.get_divergence(q);
        vector div_term   = -0.5 * divergence * u;
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
        integrator.submit_value(div_term, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  do_face_integral_nonlinear_operator(IntegratorFace & integrator_m,
                                      IntegratorFace & integrator_p) const
  {
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector normal = integrator_m.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux(uM, uP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal       = kernel.calculate_upwind_flux(uM, uP, normal);
        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector jump   = uM - uP;
        vector normal = integrator_m.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux(uM, uP, normal);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        vector flux_m = flux + 0.25 * jump * normal * uP;
        vector flux_p = -flux + 0.25 * jump * normal * uM;

        integrator_m.submit_value(flux_m, q);
        integrator_p.submit_value(flux_p, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  do_boundary_integral_nonlinear_operator(IntegratorFace &           integrator,
                                          types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       integrator,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        vector normalM = integrator.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux, uM * normalM);

        integrator.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       integrator,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal = kernel.calculate_upwind_flux(uM, uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM      = integrator.get_value(q);
        vector uP      = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       integrator,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);
        vector normalM = integrator.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux, uM * normalM);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        flux = flux + 0.25 * (uM - uP) * normalM * uP;
        integrator.submit_value(flux, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      this->integrator->reinit(cell);

      this->integrator->gather_evaluate(src,
                                        integrator_flags_linear_transport.cell_evaluate.value,
                                        integrator_flags_linear_transport.cell_evaluate.gradient,
                                        integrator_flags_linear_transport.cell_evaluate.hessian);

      // Note that the integrator flags which are valid here are different from those for the
      // linearized operator!
      integrator_velocity->reinit(cell);
      integrator_velocity->gather_evaluate(velocity, true, false, false);

      do_cell_integral_linear_transport(*this->integrator);

      this->integrator->integrate_scatter(integrator_flags_linear_transport.cell_integrate.value,
                                          integrator_flags_linear_transport.cell_integrate.gradient,
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

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);
      this->integrator_p->reinit(face);

      this->integrator_m->gather_evaluate(src,
                                          integrator_flags_linear_transport.face_evaluate.value,
                                          integrator_flags_linear_transport.face_evaluate.gradient);

      this->integrator_p->gather_evaluate(src,
                                          integrator_flags_linear_transport.face_evaluate.value,
                                          integrator_flags_linear_transport.face_evaluate.gradient);

      integrator_velocity_m->reinit(face);
      integrator_velocity_m->gather_evaluate(velocity, true, false);

      integrator_velocity_p->reinit(face);
      integrator_velocity_p->gather_evaluate(velocity, true, false);

      do_face_integral_linear_transport(*this->integrator_m, *this->integrator_p);

      this->integrator_m->integrate_scatter(
        integrator_flags_linear_transport.face_integrate.value,
        integrator_flags_linear_transport.face_integrate.gradient,
        dst);

      this->integrator_p->integrate_scatter(
        integrator_flags_linear_transport.face_integrate.value,
        integrator_flags_linear_transport.face_integrate.gradient,
        dst);
    }
  }

  void
  boundary_face_loop_linear_transport(MatrixFree<dim, Number> const & matrix_free,
                                      VectorType &                    dst,
                                      VectorType const &              src,
                                      Range const &                   face_range) const
  {
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      this->integrator_m->reinit(face);
      this->integrator_m->gather_evaluate(src,
                                          integrator_flags_linear_transport.face_evaluate.value,
                                          integrator_flags_linear_transport.face_evaluate.gradient);

      integrator_velocity_m->reinit(face);
      integrator_velocity_m->gather_evaluate(velocity, true, false);

      do_boundary_integral_linear_transport(*this->integrator_m, matrix_free.get_boundary_id(face));

      this->integrator_m->integrate_scatter(
        integrator_flags_linear_transport.face_integrate.value,
        integrator_flags_linear_transport.face_integrate.gradient,
        dst);
    }
  }

  void
  do_cell_integral_linear_transport(IntegratorCell & integrator) const
  {
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F = uw
        vector u = integrator.get_value(q);
        vector w = integrator_velocity->get_value(q);
        tensor F = outer_product(u, w);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective formulation: grad(u) * w
        vector w      = integrator_velocity->get_value(q);
        tensor grad_u = integrator.get_gradient(q);
        vector F      = grad_u * w;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  do_face_integral_linear_transport(IntegratorFace & integrator_m,
                                    IntegratorFace & integrator_p) const
  {
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_m.get_value(q);
        vector uP = integrator_p.get_value(q);

        vector wM = integrator_velocity_m->get_value(q);
        vector wP = integrator_velocity_p->get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_m.get_value(q);
        vector uP = integrator_p.get_value(q);

        vector wM = integrator_velocity_m->get_value(q);
        vector wP = integrator_velocity_p->get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linear_transport(uM, uP, wM, wP, normal);

        scalar average_normal_velocity = 0.5 * (wM + wP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  do_boundary_integral_linear_transport(IntegratorFace &           integrator,
                                        types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       integrator,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        // concerning the transport velocity w, use the same value for interior and
        // exterior states, i.e., do not prescribe boundary conditions
        vector wM = integrator_velocity_m->get_value(q);

        vector normal = integrator.get_normal_vector(q);

        vector flux = kernel.calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wM, normal);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux, wM * normal);

        integrator.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       integrator,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        // concerning the transport velocity w, use the same value for interior and
        // exterior states, i.e., do not prescribe boundary conditions
        vector wM = integrator_velocity_m->get_value(q);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linear_transport(uM, uP, wM, wM, normal);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux_times_normal, wM * normal);

        scalar average_normal_velocity = wM * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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

    // TODO -> shift to kernel
    integrator_velocity->reinit(cell);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
      integrator_velocity->gather_evaluate(velocity, true, false, false);
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
      integrator_velocity->gather_evaluate(velocity, true, true, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face(unsigned int const face) const
  {
    Base::reinit_face(face);

    // TODO -> shift to kernel
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(velocity, true, false);
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_boundary_face(unsigned int const face) const
  {
    Base::reinit_boundary_face(face);

    // TODO -> shift to kernel
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);
  }

  // Note: this function can only be used for the linearized operator.
  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const
  {
    Base::reinit_face_cell_based(cell, face, boundary_id);

    // TODO -> shift to kernel
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

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector delta_u = integrator.get_value(q);
        vector u       = integrator_velocity->get_value(q);
        tensor F       = outer_product(u, delta_u);

        // minus sign due to integration by parts
        integrator.submit_gradient(-(F + transpose(F)), q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective term: grad(u) * u
        vector u            = integrator_velocity->get_value(q);
        tensor grad_u       = integrator_velocity->get_gradient(q);
        vector delta_u      = integrator.get_value(q);
        tensor grad_delta_u = integrator.get_gradient(q);

        vector F = grad_u * delta_u + grad_delta_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP = integrator_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux =
          kernel.calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = -n⁻
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP = integrator_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                    delta_average_normal_velocity * uM,
                                  q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * delta_uP +
                                    delta_average_normal_velocity * uP,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal_m = integrator_m.get_normal_vector(q);

        vector flux =
          kernel.calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal_m);

        integrator_m.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                    delta_average_normal_velocity * uM,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        // TODO
        // Accessing exterior data is currently not available in deal.II/matrixfree.
        // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
        // are not calculated exactly.
        vector uP = uM;

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal_m = integrator_m.get_normal_vector(q);

        vector flux =
          kernel.calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal_m);

        integrator_m.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        // TODO
        // Accessing exterior data is currently not available in deal.II/matrixfree.
        // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
        // are not calculated exactly.
        vector uP = uM;

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                    delta_average_normal_velocity * uM,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_m;

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = integrator_p.get_value(q);

        vector normal_p = -integrator_p.get_normal_vector(q);

        vector flux =
          kernel.calculate_lax_friedrichs_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        integrator_p.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = integrator_velocity_p->get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = integrator_p.get_value(q);

        vector normal_p = -integrator_p.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal_p;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal_p;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(flux_times_normal - average_normal_velocity * delta_uP -
                                    delta_average_normal_velocity * uP,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       *integrator_velocity_m,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        vector delta_uM = integrator.get_value(q);
        vector delta_uP = calculate_exterior_value_linearized(
          delta_uM, q, integrator, boundary_type, this->operator_data.type_dirichlet_bc);

        vector normal = integrator.get_normal_vector(q);

        vector flux =
          kernel.calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux, uM * normal);

        integrator.submit_value(flux, q);
      }
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator_velocity_m->get_value(q);
        vector uP = calculate_exterior_value_nonlinear(uM,
                                                       q,
                                                       *integrator_velocity_m,
                                                       boundary_type,
                                                       boundary_id,
                                                       this->operator_data.bc,
                                                       this->eval_time,
                                                       this->operator_data.type_dirichlet_bc);

        vector delta_uM = integrator.get_value(q);
        vector delta_uP = calculate_exterior_value_linearized(
          delta_uM, q, integrator, boundary_type, this->operator_data.type_dirichlet_bc);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal =
          kernel.calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && this->operator_data.use_outflow_bc == true)
          kernel.apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                  delta_average_normal_velocity * uM,
                                q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  mutable IntegratorFlags integrator_flags_linear_transport;

  // TODO shift to ConvectiveKernel
  mutable VectorType velocity;

  mutable std::shared_ptr<IntegratorCell> integrator_velocity;
  mutable std::shared_ptr<IntegratorFace> integrator_velocity_m;
  mutable std::shared_ptr<IntegratorFace> integrator_velocity_p;
  // TODO shift to ConvectiveKernel

  Operators::ConvectiveKernel<dim, Number> kernel;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
