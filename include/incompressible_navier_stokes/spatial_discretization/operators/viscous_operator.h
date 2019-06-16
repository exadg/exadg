/*
 * viscous_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/interior_penalty_parameter.h"
#include "../../../operators/operator_base.h"
#include "../../../operators/variable_coefficients.h"
#include "../../user_interface/input_parameters.h"
#include "weak_boundary_conditions.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
struct ViscousKernelData
{
  ViscousKernelData()
    : IP_factor(1.0),
      degree(1),
      degree_mapping(1),
      viscosity(1.0),
      formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
      IP_formulation(InteriorPenaltyFormulation::SIPG),
      viscosity_is_variable(false)
  {
  }

  double                           IP_factor;
  unsigned int                     degree;
  unsigned int                     degree_mapping;
  double                           viscosity;
  FormulationViscousTerm           formulation_viscous_term;
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;
  InteriorPenaltyFormulation       IP_formulation;
  bool                             viscosity_is_variable;
};

template<int dim, typename Number>
class ViscousKernel
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         ViscousKernelData const &       data_in,
         unsigned int const              dof_index) const
  {
    data = data_in;

    MappingQGeneric<dim> mapping(data_in.degree_mapping);
    IP::calculate_penalty_parameter<dim, Number>(
      array_penalty_parameter, matrix_free, mapping, data_in.degree, dof_index);

    AssertThrow(data.viscosity >= 0.0, ExcMessage("Viscosity is not set!"));
  }

  void
  set_viscosity_coefficients_ptr(std::shared_ptr<VariableCoefficients<dim, Number>> coefficients)
  {
    viscosity_coefficients = coefficients;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    flags.face_evaluate  = FaceFlags(true, true);
    flags.face_integrate = FaceFlags(true, true);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells       = update_JxW_values | update_gradients;
    flags.inner_faces = update_JxW_values | update_gradients | update_normal_vectors;
    flags.boundary_faces =
      update_JxW_values | update_gradients | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
  }

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const
  {
    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
    }
  }

  /*
   * This functions return the viscosity for a given cell in a given quadrature point
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_cell(unsigned int const cell, unsigned int const q) const
  {
    scalar viscosity = make_vectorized_array<Number>(data.viscosity);

    if(data.viscosity_is_variable)
    {
      viscosity = viscosity_coefficients->get_coefficient_cell(cell, q);
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
    scalar average_viscosity = make_vectorized_array<Number>(0.0);

    scalar coefficient_face = viscosity_coefficients->get_coefficient_face(face, q);
    scalar coefficient_face_neighbor =
      viscosity_coefficients->get_coefficient_face_neighbor(face, q);

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
    scalar viscosity = make_vectorized_array<Number>(data.viscosity);

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
    scalar viscosity = make_vectorized_array<Number>(data.viscosity);

    if(data.viscosity_is_variable)
    {
      viscosity = viscosity_coefficients->get_coefficient_face(face, q);
    }

    return viscosity;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    get_volume_flux(IntegratorCell const & integrator, unsigned int const q) const
  {
    scalar viscosity = get_viscosity_cell(integrator.get_cell_index(), q);

    if(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      return viscosity * make_vectorized_array<Number>(2.) * integrator.get_symmetric_gradient(q);
    }
    else if(data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      return viscosity * integrator.get_gradient(q);
    }
    else
    {
      AssertThrow(data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                    data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));

      return tensor();
    }
  }

  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    calculate_value_flux(vector const & value_m,
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
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
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
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified formulation of viscous term is not implemented."));
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
      gradient = make_vectorized_array<Number>(2.0) * integrator.get_symmetric_gradient(q);
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
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    vector normal_gradient = gradient * integrator.get_normal_vector(q);

    return normal_gradient;
  }

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_gradient_flux(vector const & normal_gradient_m,
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
        AssertThrow(data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::Symmetrized ||
                      data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::NotSymmetrized,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
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
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return gradient_flux;
  }

  // clang-format off
  /*
   *  These two functions calculates the velocity gradient in normal
   *  direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *  Divergence formulation: F(u) = nu * ( grad(u) + grad(u)^T )
   *  Laplace formulation: F(u) = nu * grad(u)
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n + 2h               | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n                    | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | F(u⁺)*n = F(u⁻)*n, F(u⁻)*n = 0  | F(u⁺)*n = -F(u⁻)*n + 2h , F(u⁻)*n = 0 | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n, F(u⁻)*n = 0 |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = h                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = 0                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | {{F(u)}}*n = 0                  | {{F(u)}}*n = h                        | {{F(u)}}*n = 0                                     |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
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
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

private:
  mutable ViscousKernelData data;

  mutable AlignedVector<scalar> array_penalty_parameter;
  mutable scalar                tau;

  std::shared_ptr<VariableCoefficients<dim, Number>> viscosity_coefficients;
};

} // namespace Operators

template<int dim>
struct ViscousOperatorData : public OperatorBaseData
{
  ViscousOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::ViscousKernelData kernel_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, typename Number>
class ViscousOperator : public OperatorBase<dim, Number, ViscousOperatorData<dim>, dim>
{
public:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef OperatorBase<dim, Number, ViscousOperatorData<dim>, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         ViscousOperatorData<dim> const &  operator_data) const;

  void
  set_viscosity_coefficients_ptr(std::shared_ptr<VariableCoefficients<dim, Number>> coefficients);

private:
  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  Operators::ViscousKernel<dim, Number> kernel;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_ \
        */
