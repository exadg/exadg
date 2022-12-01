/*
 * viscous_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
namespace IncNS
{
namespace Operators
{
struct ViscousKernelData
{
  ViscousKernelData()
    : IP_factor(1.0),
      viscosity(1.0),
      formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
      IP_formulation(InteriorPenaltyFormulation::SIPG),
      viscosity_is_variable(false),
      variable_normal_vector(false)
  {
  }

  double                           IP_factor;
  double                           viscosity;
  FormulationViscousTerm           formulation_viscous_term;
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;
  InteriorPenaltyFormulation       IP_formulation;
  bool                             viscosity_is_variable;
  bool                             variable_normal_vector;
};

template<int dim, typename Number>
class ViscousKernel
{
private:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  ViscousKernel() : degree(1), tau(dealii::make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         ViscousKernelData const &               data,
         unsigned int const                      dof_index)
  {
    this->data = data;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    AssertThrow(data.viscosity >= 0.0, dealii::ExcMessage("Viscosity is not set!"));

    if(data.viscosity_is_variable)
    {
      // allocate vectors for variable coefficients and initialize with constant viscosity
      viscosity_coefficients.initialize(matrix_free, degree, data.viscosity);
    }
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
  }

  ViscousKernelData const &
  get_data() const
  {
    return this->data;
  }

  void
  set_coefficient_cell(unsigned int const cell, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_cell(cell, q, value);
  }

  scalar
  get_coefficient_face(unsigned int const face, unsigned int const q)
  {
    return viscosity_coefficients.get_coefficient_face(face, q);
  }

  void
  set_coefficient_face(unsigned int const face, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_face(face, q, value);
  }

  void
  set_coefficient_face_neighbor(unsigned int const face, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_face_neighbor(face, q, value);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values | dealii::update_gradients;
    if(compute_interior_face_integrals)
      flags.inner_faces =
        dealii::update_JxW_values | dealii::update_gradients | dealii::update_normal_vectors;
    if(compute_boundary_face_integrals)
      flags.boundary_faces = dealii::update_JxW_values | dealii::update_gradients |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(degree, data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(degree, data.IP_factor);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<Number>(degree, data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<Number>(degree, data.IP_factor);
    }
  }

  /*
   * This functions return the viscosity for a given cell in a given quadrature point
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_cell(unsigned int const cell, unsigned int const q) const
  {
    scalar viscosity = dealii::make_vectorized_array<Number>(data.viscosity);

    if(data.viscosity_is_variable)
    {
      viscosity = viscosity_coefficients.get_coefficient_cell(cell, q);
    }

    return viscosity;
  }

  /*
   *  This function calculates the average viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_average_viscosity(unsigned int const face, unsigned int const q) const
  {
    scalar average_viscosity = dealii::make_vectorized_array<Number>(0.0);

    scalar coefficient_face = viscosity_coefficients.get_coefficient_face(face, q);
    scalar coefficient_face_neighbor =
      viscosity_coefficients.get_coefficient_face_neighbor(face, q);

    // harmonic mean (harmonic weighting according to Schott and Rasthofer et al. (2015))
    average_viscosity = 2.0 * coefficient_face * coefficient_face_neighbor /
                        (coefficient_face + coefficient_face_neighbor);

    // arithmetic mean
    // average_viscosity = 0.5 * (coefficient_face + coefficient_face_neighbor);

    // maximum value
    // average_viscosity = std::max(coefficient_face, coefficient_face_neighbor);

    return average_viscosity;
  }

  /*
   *  This function returns the viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_interior_face(unsigned int const face, unsigned int const q) const
  {
    scalar viscosity = dealii::make_vectorized_array<Number>(data.viscosity);

    if(data.viscosity_is_variable)
    {
      viscosity = calculate_average_viscosity(face, q);
    }

    return viscosity;
  }

  /*
   *  This function returns the viscosity for boundary faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_boundary_face(unsigned int const face, unsigned int const q) const
  {
    scalar viscosity = dealii::make_vectorized_array<Number>(data.viscosity);

    if(data.viscosity_is_variable)
    {
      viscosity = viscosity_coefficients.get_coefficient_face(face, q);
    }

    return viscosity;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_volume_flux(tensor const & gradient, scalar const & viscosity) const
  {
    if(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      return viscosity * (gradient + transpose(gradient));
    }
    else if(data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      return viscosity * gradient;
    }
    else
    {
      AssertThrow(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                    data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented."));

      return tensor();
    }
  }

  /*
   *  Calculation of "gradient_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    calculate_gradient_flux(vector const & value_m,
                            vector const & value_p,
                            vector const & normal,
                            scalar const & viscosity) const
  {
    tensor value_flux;

    vector jump_value  = value_m - value_p;
    tensor jump_tensor = outer_product(jump_value, normal);

    if(data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      if(data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * jump_tensor;
      }
      else if(data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * jump_tensor;
      }
      else
      {
        AssertThrow(
          false, dealii::ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else if(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else if(data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else
      {
        AssertThrow(
          false, dealii::ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return value_flux;
  }

  /*
   *  This function calculates the gradient in normal direction on element e
   *  depending on the formulation of the viscous term.
   */
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_normal_gradient(unsigned int const q, Integrator & integrator) const
  {
    tensor gradient;

    if(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * F = 2 * nu * symmetric_gradient
       *   = 2.0 * nu * 1/2 (grad(u) + grad(u)^T)
       */
      gradient = dealii::make_vectorized_array<Number>(2.0) * integrator.get_symmetric_gradient(q);
    }
    else if(data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       *  F = nu * grad(u)
       */
      gradient = integrator.get_gradient(q);
    }
    else
    {
      AssertThrow(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                    data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    vector normal_gradient = gradient * integrator.get_normal_vector(q);

    return normal_gradient;
  }

  /*
   *  Calculation of "value_flux". Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_value_flux(vector const & normal_gradient_m,
                         vector const & normal_gradient_p,
                         vector const & value_m,
                         vector const & value_p,
                         vector const & normal,
                         scalar const & viscosity) const
  {
    vector gradient_flux;

    vector jump_value              = value_m - value_p;
    vector average_normal_gradient = 0.5 * (normal_gradient_m + normal_gradient_p);

    if(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::Symmetrized)
      {
        gradient_flux = viscosity * average_normal_gradient -
                        viscosity * tau * (jump_value + (jump_value * normal) * normal);
      }
      else if(data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::NotSymmetrized)
      {
        gradient_flux = viscosity * average_normal_gradient - viscosity * tau * jump_value;
      }
      else
      {
        AssertThrow(
          data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::Symmetrized ||
            data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::NotSymmetrized,
          dealii::ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
    else if(data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      gradient_flux = viscosity * average_normal_gradient - viscosity * tau * jump_value;
    }
    else
    {
      AssertThrow(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                    data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return gradient_flux;
  }

  // clang-format off
  /*
   *  This function calculates the interior velocity gradient in normal
   *  direction depending on the operator type.
   *
   *  The variable normal_gradient_m has the meaning of F(u⁻)*n with
   *
   *  Divergence formulation: F(u) = F_nu(u) / nu = nu * ( grad(u) + grad(u)^T )
   *  Laplace formulation:    F(u) = F_nu(u) / nu = nu * grad(u)
   *
   */
  // clang-format on
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_normal_gradient(unsigned int const   q,
                                       Integrator const &   integrator,
                                       OperatorType const & operator_type) const
  {
    vector normal_gradient_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      normal_gradient_m = calculate_normal_gradient(q, integrator);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, normal_gradient_m is already initialized with 0
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

private:
  ViscousKernelData data;

  unsigned int degree;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;

  VariableCoefficients<dim, Number> viscosity_coefficients;
};

} // namespace Operators

template<int dim>
struct ViscousOperatorData : public OperatorBaseData
{
  ViscousOperatorData() : OperatorBaseData()
  {
  }

  Operators::ViscousKernelData kernel_data;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};

template<int dim, typename Number>
class ViscousOperator : public OperatorBase<dim, Number, dim>
{
public:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef OperatorBase<dim, Number, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free,
             dealii::AffineConstraints<Number> const &              affine_constraints,
             ViscousOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel);

  void
  update();

private:
  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const;

  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  ViscousOperatorData<dim> operator_data;

  std::shared_ptr<Operators::ViscousKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_ \
        */
