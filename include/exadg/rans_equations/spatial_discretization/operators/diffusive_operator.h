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

#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include <exadg/rans_equations/user_interface/boundary_descriptor.h>
#include <exadg/rans_equations/user_interface/parameters.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>

#include <exadg/rans_equations/spatial_discretization/turbulence_model.h>
#include <exadg/rans_equations/user_interface/viscosity_model_data.h>


namespace ExaDG
{
namespace RANS
{
namespace Operators
{
struct DiffusiveKernelData
{
  DiffusiveKernelData() : IP_factor(1.0),
    diffusivity(1.0),
    turbulence_model_enabled(false),
    scalar_type(ScalarType::Scalar),
    positivity_preserving_limiter(PositivityPreservingLimiter::Undefined)
  {
  }

  double IP_factor;
  double diffusivity;
  bool turbulence_model_enabled;
  ScalarType scalar_type;
  TurbulenceModelData turbulence_model_data;
  PositivityPreservingLimiter positivity_preserving_limiter;
};

template<int dim, typename Number>
class DiffusiveKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  DiffusiveKernel() : degree(1), tau(dealii::make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         DiffusiveKernelData const &             data_in,
         unsigned int const                      dof_index)
  {
    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    AssertThrow(data.diffusivity > (0.0 - std::numeric_limits<double>::epsilon()),
                dealii::ExcMessage("Diffusivity is not set!"));
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
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

    flags.cells = dealii::update_gradients | dealii::update_JxW_values;
    if(compute_interior_face_integrals)
      flags.inner_faces =
        dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
    if(compute_boundary_face_integrals)
      flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const dof_index) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
  }


  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_gradient_flux(scalar const & value_m,
                            scalar const & value_p,
                            scalar const & viscosity) const
  {
    return -0.5 * viscosity * (value_m - value_p);
  }

  /*
   * Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since the
   * flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal, where
   * normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & normal_gradient_m,
                         scalar const & normal_gradient_p,
                         scalar const & value_m,
                         scalar const & value_p,
                         scalar const & viscosity) const
  {
    return viscosity *
           (0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p));
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral along with grad of shape function
   */
  inline DEAL_II_ALWAYS_INLINE //
  vector
  get_grad_volume_flux(IntegratorCell & integrator,
                       scalar const & viscosity,
                       unsigned int const q) const
  {
    vector result;
    if(data.turbulence_model_enabled)
    {
      result = sipg_cell_integral(integrator.get_gradient(q), viscosity);
    }
    else {
      result = integrator.get_gradient(q) * data.diffusivity;
    }
    return result;
  }

  /*
   * \left( \nu + \frac{\nu_{T}}{\sigma_k} * grad(k) \right)
  */
  vector
  sipg_cell_integral(vector const & sol_grad,
                     scalar const & viscosity) const
  {
    return sol_grad * viscosity;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral along with value of shape function
   */
  inline DEAL_II_ALWAYS_INLINE //
  scalar
  get_value_volume_flux(IntegratorCell & integrator,
                        scalar const & eddy_viscosity,
                        scalar const & effective_viscosity,
                        unsigned int const q) const
  {
    scalar result = dealii::make_vectorized_array<Number>(0.);
    vector sol_grad = integrator.get_gradient(q);
    scalar sol = integrator.get_value(q);
    /*result = sipg_varying_viscosity_integral(sol, sol_grad, eddy_viscosity);*/
    /*if(data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable)*/
    /*{*/
    /*  result += limiter_term(sol_grad, effective_viscosity);*/
    /*}*/
    return result;
  }

  /*
   * \frac{1}{\sigma_{k}} \frac{1}{2 k^{1/2} \ell} e^{\kappa} \frac{\partial \kappa}{\partial x_j}\frac{\partial \kappa}{\partial x_j}
   */
  inline DEAL_II_ALWAYS_INLINE
  scalar
  sipg_varying_viscosity_integral(scalar const & sol,
                                  vector const & sol_grad,
                                  scalar const & viscosity) const
  {
    scalar value = dealii::make_vectorized_array<Number>(0.0);

    if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
      if(turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength)
      {
        value = dealii::make_vectorized_array<Number>(1/(2.0*turbulence_model_ptr->model_coefficients[0])) * viscosity * sol_grad.norm_square();
      }
      else if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        if (turbulence_model_ptr->scalar_type==ScalarType::TurbulentKineticEnergy) {
          value = dealii::make_vectorized_array<Number>(2.0 / turbulence_model_ptr->model_coefficients[0]) * sol_grad.norm_square() * viscosity;
        }
        else if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
          value = dealii::make_vectorized_array<Number>(-1.0/turbulence_model_ptr->model_coefficients[4]) * sol_grad.norm_square() * viscosity;
        }
      }
      else {
        AssertThrow(false, dealii::ExcMessage(" Positivity Limiter only available for TurbulenceEddyViscosityModel::PrandtlMixingLength and TurbulenceEddyViscosityModel::StandardKEpsilon"));
      }
    }
    else if (data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper) {
      if(turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength)
      {
        value = dealii::make_vectorized_array<Number>(1/(2.0*turbulence_model_ptr->model_coefficients[0])) * viscosity * sol_grad.norm_square();
      }
      else if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        if (turbulence_model_ptr->scalar_type==ScalarType::TurbulentKineticEnergy) {
          value = dealii::make_vectorized_array<Number>(2.0 / turbulence_model_ptr->model_coefficients[0]) * sol_grad.norm_square() * viscosity / sol;
        }
        else if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
          value = dealii::make_vectorized_array<Number>(-1.0/turbulence_model_ptr->model_coefficients[4]) * sol_grad.norm_square() * viscosity / sol;
        }
      }
      else {
        AssertThrow(false, dealii::ExcMessage(" Positivity Limiter only available for TurbulenceEddyViscosityModel::PrandtlMixingLength and TurbulenceEddyViscosityModel::StandardKEpsilon"));
      }
    }

    return value;
  }

  /*
   *\left( \nu + \frac{\nu_{T}}{\sigma_k} grad(k) grad(k) \right)
  */
  scalar
  limiter_term(vector const & sol_grad,
               scalar const & viscosity) const
  {
    scalar value = dealii::make_vectorized_array<Number>(0.0);
    value = viscosity * sol_grad.norm_square();
    /*std::cout << "Positivity limiter term : " << value << std::endl;*/
    return value;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    return integrator.get_gradient(q) * data.diffusivity;
  }

  DiffusiveKernelData data;

  mutable std::shared_ptr<TurbulenceModel<dim, Number>> turbulence_model_ptr;
private:

  unsigned int degree;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators


template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData
{
  DiffusiveOperatorData() : OperatorBaseData()
  {
  }

  Operators::DiffusiveKernelData kernel_data;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};


template<int dim, typename Number>
class DiffusiveOperator : public OperatorBase<dim, Number, 1>
{
private:
  typedef OperatorBase<dim, Number, 1> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &                  matrix_free,
             dealii::AffineConstraints<Number> const &                affine_constraints,
             DiffusiveOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::DiffusiveKernel<dim, Number>> kernel);

  void
  update();

private:
  void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const final;

  void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const final;

  void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  DiffusiveOperatorData<dim> operator_data;

  std::shared_ptr<Operators::DiffusiveKernel<dim, Number>> kernel;
};
} // namespace RANS
} // namespace ExaDG

#endif
