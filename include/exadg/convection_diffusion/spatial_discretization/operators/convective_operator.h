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

#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_base.h>

#include <memory>

namespace ExaDG
{
namespace ConvDiff
{
namespace Operators
{
template<int dim>
struct ConvectiveKernelData
{
  ConvectiveKernelData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      velocity_type(TypeVelocityField::Function),
      dof_index_velocity(1),
      numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined)
  {
  }

  // formulation
  FormulationConvectiveTerm formulation;

  // analytical vs. numerical velocity field
  TypeVelocityField velocity_type;

  // TypeVelocityField::DoFVector
  unsigned int dof_index_velocity;

  // TypeVelocityField::Function
  std::shared_ptr<dealii::Function<dim>> velocity;

  // numerical flux (e.g., central flux vs. Lax-Friedrichs flux)
  NumericalFluxConvectiveOperator numerical_flux_formulation;
};

template<int dim, typename Number>
class ConvectiveKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVelocity;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorVelocity;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         ConvectiveKernelData<dim> const &       data_in,
         unsigned int const                      quad_index,
         bool const                              use_own_velocity_storage)
  {
    data = data_in;

    if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      integrator_velocity =
        std::make_shared<CellIntegratorVelocity>(matrix_free, data.dof_index_velocity, quad_index);

      integrator_velocity_m = std::make_shared<FaceIntegratorVelocity>(matrix_free,
                                                                       true,
                                                                       data.dof_index_velocity,
                                                                       quad_index);

      integrator_velocity_p = std::make_shared<FaceIntegratorVelocity>(matrix_free,
                                                                       false,
                                                                       data.dof_index_velocity,
                                                                       quad_index);

      if(use_own_velocity_storage)
      {
        velocity.reset();
        matrix_free.initialize_dof_vector(velocity.own(), data.dof_index_velocity);
      }
    }
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      flags.cell_evaluate  = dealii::EvaluationFlags::values;
      flags.cell_integrate = dealii::EvaluationFlags::gradients;
    }
    else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
      flags.cell_integrate = dealii::EvaluationFlags::values;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    flags.face_evaluate  = dealii::EvaluationFlags::values;
    flags.face_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_gradients | dealii::update_JxW_values |
                  dealii::update_quadrature_points; // q-points due to analytical velocity field
    flags.inner_faces = dealii::update_JxW_values | dealii::update_quadrature_points |
                        dealii::update_normal_vectors; // q-points due to analytical velocity field
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_normal_vectors;

    return flags;
  }

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const
  {
    return *velocity;
  }

  void
  set_velocity_copy(VectorType const & velocity_in) const
  {
    AssertThrow(data.velocity_type == TypeVelocityField::DoFVector,
                dealii::ExcMessage("Invalid parameter velocity_type."));

    velocity.own() = velocity_in;

    velocity->update_ghost_values();
  }

  void
  set_velocity_ptr(VectorType const & velocity_in) const
  {
    AssertThrow(data.velocity_type == TypeVelocityField::DoFVector,
                dealii::ExcMessage("Invalid parameter velocity_type."));

    velocity.reset(velocity_in);

    velocity->update_ghost_values();
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      integrator_velocity->reinit(cell);
      integrator_velocity->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face(unsigned int const face) const
  {
    if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      integrator_velocity_m->reinit(face);
      integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

      integrator_velocity_p->reinit(face);
      integrator_velocity_p->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      integrator_velocity_m->reinit(face);
      integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const
  {
    if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      integrator_velocity_m->reinit(cell, face);
      integrator_velocity_m->gather_evaluate(*velocity, dealii::EvaluationFlags::values);

      if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
      {
        // TODO: Matrix-free implementation in deal.II does currently not allow to access data of
        // the neighboring element in case of cell-based face loops.
        //      integrator_velocity_p->reinit(cell, face);
        //      integrator_velocity_p->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
      }
    }
  }

  /*
   * This function calculates the numerical flux using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_central_flux(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity) const
  {
    scalar average_value = 0.5 * (value_m + value_p);

    return normal_velocity * average_value;
  }

  /*
   * The same as above, but with discontinuous velocity field.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_central_flux(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity_m,
                           scalar const & normal_velocity_p) const
  {
    return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p);
  }

  /*
   * This function calculates the numerical flux using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux(scalar const & value_m,
                                  scalar const & value_p,
                                  scalar const & normal_velocity) const
  {
    scalar average_value = 0.5 * (value_m + value_p);
    scalar jump_value    = value_m - value_p;
    scalar lambda        = std::abs(normal_velocity);

    return normal_velocity * average_value + 0.5 * lambda * jump_value;
  }

  /*
   * The same as above, but with discontinuous velocity field.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux(scalar const & value_m,
                                  scalar const & value_p,
                                  scalar const & normal_velocity_m,
                                  scalar const & normal_velocity_p) const
  {
    scalar jump_value = value_m - value_p;
    scalar lambda     = std::max(std::abs(normal_velocity_m), std::abs(normal_velocity_p));

    return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p) +
           0.5 * lambda * jump_value;
  }

  /*
   * This function calculates the numerical flux where the type of the numerical flux depends on the
   * specified parameter. This function handles both analytical and numerical velocity fields.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(unsigned int const q,
                   IntegratorFace &   integrator,
                   scalar const &     value_m,
                   scalar const &     value_p,
                   vector const &     normal_m,
                   Number const &     time,
                   bool const         exterior_velocity_available) const
  {
    scalar flux = dealii::make_vectorized_array<Number>(0.0);

    if(data.velocity_type == TypeVelocityField::Function)
    {
      dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

      vector velocity = FunctionEvaluator<1, dim, Number>::value(*(data.velocity), q_points, time);

      scalar normal_velocity = velocity * normal_m;

      // flux functions are the same for DivergenceFormulation and ConvectiveFormulation

      if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
      {
        flux = calculate_central_flux(value_m, value_p, normal_velocity);
      }
      else if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
      {
        flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity);
      }
    }
    else if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      vector velocity_m = integrator_velocity_m->get_value(q);
      vector velocity_p =
        exterior_velocity_available ? integrator_velocity_p->get_value(q) : velocity_m;

      scalar normal_velocity_m = velocity_m * normal_m;
      scalar normal_velocity_p = velocity_p * normal_m;

      if(data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
      {
        if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
        {
          flux = calculate_central_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
        }
        else if(data.numerical_flux_formulation ==
                NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
        {
          flux =
            calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
        }
      }
      else if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
      {
        scalar normal_velocity = 0.5 * (normal_velocity_m + normal_velocity_p);

        if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
        {
          flux = calculate_central_flux(value_m, value_p, normal_velocity);
        }
        else if(data.numerical_flux_formulation ==
                NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
        {
          flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity);
        }
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return flux;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_average_velocity(unsigned int const q,
                               IntegratorFace &   integrator,
                               Number const &     time,
                               bool const         exterior_velocity_available) const
  {
    vector velocity;

    if(data.velocity_type == TypeVelocityField::Function)
    {
      dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

      velocity = FunctionEvaluator<1, dim, Number>::value(*(data.velocity), q_points, time);
    }
    else if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      vector velocity_m = integrator_velocity_m->get_value(q);
      vector velocity_p =
        exterior_velocity_available ? integrator_velocity_p->get_value(q) : velocity_m;

      velocity = 0.5 * (velocity_m + velocity_p);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return velocity;
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, scalar>
    calculate_flux_interior_and_neighbor(unsigned int const q,
                                         IntegratorFace &   integrator,
                                         scalar const &     value_m,
                                         scalar const &     value_p,
                                         vector const &     normal_m,
                                         Number const &     time,
                                         bool const         exterior_velocity_available) const
  {
    scalar fluxM =
      calculate_flux(q, integrator, value_m, value_p, normal_m, time, exterior_velocity_available);
    scalar fluxP = -fluxM;

    if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector velocity =
        calculate_average_velocity(q, integrator, time, exterior_velocity_available);
      scalar normal_velocity = velocity * normal_m;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      fluxM = fluxM - normal_velocity * value_m;
      // opposite signs since n⁺ = - n⁻
      fluxP = fluxP + normal_velocity * value_p;
    }

    return std::make_tuple(fluxM, fluxP);
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux_interior(unsigned int const q,
                            IntegratorFace &   integrator,
                            scalar const &     value_m,
                            scalar const &     value_p,
                            vector const &     normal_m,
                            Number const &     time,
                            bool const         exterior_velocity_available) const
  {
    scalar flux =
      calculate_flux(q, integrator, value_m, value_p, normal_m, time, exterior_velocity_available);

    if(data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector velocity =
        calculate_average_velocity(q, integrator, time, exterior_velocity_available);
      scalar normal_velocity = velocity * normal_m;

      // second term appears since the strong formulation is implemented (integration by parts
      // is performed twice)
      flux = flux - normal_velocity * value_m;
    }

    return flux;
  }


  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_divergence_form(scalar const &     value,
                                    IntegratorCell &   integrator,
                                    unsigned int const q,
                                    Number const &     time) const
  {
    vector velocity;

    if(data.velocity_type == TypeVelocityField::Function)
    {
      velocity = FunctionEvaluator<1, dim, Number>::value(*(data.velocity),
                                                          integrator.quadrature_point(q),
                                                          time);
    }
    else if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      velocity = integrator_velocity->get_value(q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return (-value * velocity);
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux_convective_form(vector const &     gradient,
                                    IntegratorCell &   integrator,
                                    unsigned int const q,
                                    Number const &     time) const
  {
    vector velocity;

    if(data.velocity_type == TypeVelocityField::Function)
    {
      velocity = FunctionEvaluator<1, dim, Number>::value(*(data.velocity),
                                                          integrator.quadrature_point(q),
                                                          time);
    }
    else if(data.velocity_type == TypeVelocityField::DoFVector)
    {
      velocity = integrator_velocity->get_value(q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    return (velocity * gradient);
  }

private:
  ConvectiveKernelData<dim> data;

  mutable lazy_ptr<VectorType> velocity;

  std::shared_ptr<CellIntegratorVelocity> integrator_velocity;
  std::shared_ptr<FaceIntegratorVelocity> integrator_velocity_m;
  std::shared_ptr<FaceIntegratorVelocity> integrator_velocity_p;
};

} // namespace Operators


template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData() : OperatorBaseData()
  {
  }

  Operators::ConvectiveKernelData<dim> kernel_data;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};

template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, 1>
{
private:
  typedef OperatorBase<dim, Number, 1> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::VectorType VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &                   matrix_free,
             dealii::AffineConstraints<Number> const &                 affine_constraints,
             ConvectiveOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel);

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  set_velocity_copy(VectorType const & velocity) const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

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

  // TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
  // cell-based face loops
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const final;

  ConvectiveOperatorData<dim> operator_data;

  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel;
};
} // namespace ConvDiff
} // namespace ExaDG

#endif
