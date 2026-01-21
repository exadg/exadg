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

#ifndef RANS_EQUATIONS_DIFFUSIVE_OPERATOR
#define RANS_EQUATIONS_DIFFUSIVE_OPERATOR

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/rans_equations/user_interface/boundary_descriptor.h>
#include <exadg/rans_equations/user_interface/parameters.h>

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
  DiffusiveKernelData()
    : IP_factor(1.0),
      diffusivity(1.0),
      turbulence_model_enabled(false),
      scalar_type(ScalarType::Scalar),
      positivity_preserving_limiter(PositivityPreservingLimiter::Undefined)
  {
  }

  double                      IP_factor;
  double                      diffusivity;
  bool                        turbulence_model_enabled;
  ScalarType                  scalar_type;
  TurbulenceModelData         turbulence_model_data;
  PositivityPreservingLimiter positivity_preserving_limiter;

  unsigned int dof_index_eddy_viscosity;
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
         unsigned int const                      dof_index,
         unsigned int const                      quad_index)
  {
    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    AssertThrow(data.diffusivity > (0.0 - std::numeric_limits<double>::epsilon()),
                dealii::ExcMessage("Diffusivity is not set!"));

    if(data.turbulence_model_enabled)
    {
      integrator_cell_eddy_viscosity =
        std::make_shared<IntegratorCell>(matrix_free, data.dof_index_eddy_viscosity, quad_index);
      integrator_face_eddy_viscosity_m = std::make_shared<IntegratorFace>(
        matrix_free, true, data.dof_index_eddy_viscosity, quad_index);
      integrator_face_eddy_viscosity_p = std::make_shared<IntegratorFace>(
        matrix_free, false, data.dof_index_eddy_viscosity, quad_index);
    }
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
  reinit_cell(unsigned const int cell) const
  {
    if(data.turbulence_model_enabled)
    {
      integrator_cell_eddy_viscosity->reinit(cell);
      integrator_cell_eddy_viscosity->gather_evaluate(*eddy_viscosity,
                                                      dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index,
              unsigned int const face) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
    if(data.turbulence_model_enabled)
    {
      integrator_face_eddy_viscosity_m->reinit(face);
      integrator_face_eddy_viscosity_p->reinit(face);
      integrator_face_eddy_viscosity_m->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
      integrator_face_eddy_viscosity_p->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_boundary_face(IntegratorFace &   integrator_m,
                       unsigned int const dof_index,
                       unsigned int const face) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
    if(data.turbulence_model_enabled)
    {
      integrator_face_eddy_viscosity_m->reinit(face);
      integrator_face_eddy_viscosity_m->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index,
                         unsigned int const               face) const
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
    if(data.turbulence_model_enabled)
    {
      integrator_face_eddy_viscosity_m->reinit(face);
      integrator_face_eddy_viscosity_m->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);

      if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
      {
        // TODO: Matrix-free implementation in deal.II does currently not allow to access data of
        // the neighboring element in case of cell-based face loops.
        //      integrator_velocity_p->reinit(cell, face);
        //      integrator_velocity_p->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
      }
    }
  }

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in)
  {
    eddy_viscosity.own() = eddy_viscosity_in;
    eddy_viscosity->update_ghost_values();
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    scalar effective_viscosity = get_effective_cell_viscosity(q);
    // scalar effective_viscosity = data.diffusivity;
    return integrator.get_gradient(q) * effective_viscosity;
  }

  inline DEAL_II_ALWAYS_INLINE scalar
  calculate_gradient_flux(scalar const & value_m,
                          scalar const & value_p,
                          scalar const & effective_viscosity_m,
                          scalar const & effective_viscosity_p) const
  {
    scalar effective_viscosity = 0.5 * (effective_viscosity_m + effective_viscosity_p);
    // scalar effective_viscosity = data.diffusivity;
    return -0.5 * effective_viscosity * (value_m - value_p);
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
                         scalar const & effective_viscosity_m,
                         scalar const & effective_viscosity_p) const
  {
    // scalar effective_viscosity = data.diffusivity;
    // scalar consistency_term =
    //   0.5 * (normal_gradient_m + normal_gradient_p);
    scalar effective_viscosity = 0.5 * (effective_viscosity_m + effective_viscosity_p);
    scalar consistency_term =
      0.5 * (normal_gradient_m * effective_viscosity_m + normal_gradient_p * effective_viscosity_p);
    scalar penalty_term = effective_viscosity * tau * (value_m - value_p);
    return consistency_term - penalty_term;
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux_penalty(scalar const & value_m,
                                 scalar const & value_p,
                                 scalar const & effective_viscosity_m,
                                 scalar const & effective_viscosity_p) const
  {
    scalar effective_viscosity = 0.5 * (effective_viscosity_m + effective_viscosity_p);
    scalar penalty_term        = effective_viscosity * tau * (value_m - value_p);
    return penalty_term;
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux_consistency(scalar const & normal_gradient_m,
                                     scalar const & normal_gradient_p,
                                     scalar const & effective_viscosity_m,
                                     scalar const & effective_viscosity_p) const
  {
    scalar effective_viscosity = 0.5 * (effective_viscosity_m + effective_viscosity_p);
    scalar consistency_term =
      0.5 * (normal_gradient_m * effective_viscosity_m + normal_gradient_p * effective_viscosity_p);
    return consistency_term;
  }

  scalar
  get_effective_cell_viscosity(unsigned const int q) const
  {
    scalar nu_t = data.diffusivity;
    if(data.scalar_type == ScalarType::TurbulentKineticEnergy)
    {
      nu_t +=
        integrator_cell_eddy_viscosity->get_value(q) *
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[0]);
    }
    else if(data.scalar_type == ScalarType::TKEDissipationRate)
    {
      nu_t +=
        integrator_cell_eddy_viscosity->get_value(q) *
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[4]);
    }
    return nu_t;
  }

  scalar
  get_int_face_eddy_viscosity(unsigned const int q) const
  {
    scalar nu = data.diffusivity;
    if(data.scalar_type == ScalarType::TurbulentKineticEnergy)
    {
      scalar inverse_sigma =
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[0]);
      nu += integrator_face_eddy_viscosity_m->get_value(q) * inverse_sigma;
    }
    else if(data.scalar_type == ScalarType::TKEDissipationRate)
    {
      scalar inverse_sigma =
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[4]);
      nu += integrator_face_eddy_viscosity_m->get_value(q) * inverse_sigma;
    }
    return nu;
  }

  scalar
  get_ext_face_eddy_viscosity(unsigned const int q) const
  {
    scalar nu = data.diffusivity;
    if(data.scalar_type == ScalarType::TurbulentKineticEnergy)
    {
      scalar inverse_sigma =
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[0]);
      nu += integrator_face_eddy_viscosity_p->get_value(q) * inverse_sigma;
    }
    else if(data.scalar_type == ScalarType::TKEDissipationRate)
    {
      scalar inverse_sigma =
        dealii::make_vectorized_array<Number>(1.0 / turbulence_model_ptr->model_coefficients[4]);
      nu += integrator_face_eddy_viscosity_p->get_value(q) * inverse_sigma;
    }
    return nu;
  }

  DiffusiveKernelData data;

  mutable std::shared_ptr<TurbulenceModel<dim, Number>> turbulence_model_ptr;

  mutable lazy_ptr<VectorType> eddy_viscosity;

  std::shared_ptr<IntegratorCell> integrator_cell_eddy_viscosity;

  std::shared_ptr<IntegratorFace> integrator_face_eddy_viscosity_m;

  std::shared_ptr<IntegratorFace> integrator_face_eddy_viscosity_p;

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

  typedef DiffusiveOperator<dim, Number> This;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef typename Base::VectorType VectorType;
  typedef typename Base::Range      Range;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &                  matrix_free,
             dealii::AffineConstraints<Number> const &                affine_constraints,
             DiffusiveOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::DiffusiveKernel<dim, Number>> kernel);

  void
  update();

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in) const;

private:
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const final;

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

  mutable unsigned int current_face_index;
};
} // namespace RANS
} // namespace ExaDG

#endif
