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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_PROJECTION_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_PROJECTION_OPERATOR_H_

#include <exadg/grid/calculate_characteristic_element_length.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  Combined divergence and continuity penalty operator: applies the operation
 *
 *   mass operator + dt * divergence penalty operator + dt * continuity penalty operator .
 *
 *  In detail
 *
 *    Mass operator: ( v_h , u_h )_Omega^e where
 *
 *    Divergence penalty operator: ( div(v_h) , tau_div * div(u_h) )_Omega^e
 *
 *    Continuity penalty operator: ( v_h , tau_conti * jump(u_h) )_dOmega^e, where
 *
 *      jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *  and
 *
 *   v_h : test function
 *   u_h : solution
 */

/*
 *  Operator data.
 */
template<int dim>
struct ProjectionOperatorData : public OperatorBaseData
{
  ProjectionOperatorData()
    : OperatorBaseData(),
      use_divergence_penalty(true),
      use_continuity_penalty(true),
      use_boundary_data(false),
      apply_penalty_terms_in_postprocessing_step(false)
  {
  }

  // specify which penalty terms are used
  bool use_divergence_penalty, use_continuity_penalty;

  bool use_boundary_data;

  bool apply_penalty_terms_in_postprocessing_step;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};

/*
 *  Continuity penalty operator:
 *
 *    ( v_h , tau_conti * jump(u_h) )_dOmega^e
 *
 *  where
 *
 *   v_h : test function
 *   u_h : solution
 *
 *   jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) with a characteristic
 *                                  element length h derived from the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */

namespace Operators
{
struct ContinuityPenaltyKernelData
{
  ContinuityPenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      which_components(ContinuityPenaltyComponents::Normal),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;
};

/*
 *  Divergence penalty operator:
 *
 *    ( div(v_h) , tau_div * div(u_h) )_Omega^e
 *
 *  where
 *
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) with a characteristic
 *                                  element length h derived from the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 *
 */

struct DivergencePenaltyKernelData
{
  DivergencePenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;
};


template<int dim, typename Number>
class ProjectionKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  ProjectionKernel() : matrix_free(nullptr), dof_index(0), quad_index(0)
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index,
         DivergencePenaltyKernelData const &     divergence_data,
         ContinuityPenaltyKernelData const &     continuity_data)
  {
    this->matrix_free = &matrix_free;

    this->dof_index  = dof_index;
    this->quad_index = quad_index;

    this->divergence_data = divergence_data;
    this->continuity_data = continuity_data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_divergence_penalty_parameter.resize(n_cells);
    array_continuity_penalty_parameter.resize(n_cells);
  }

  DivergencePenaltyKernelData
  get_divergence_data()
  {
    return this->divergence_data;
  }

  ContinuityPenaltyKernelData
  get_continuity_data()
  {
    return this->continuity_data;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    flags.face_evaluate  = dealii::EvaluationFlags::values;
    flags.face_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values | dealii::update_gradients;

    flags.inner_faces = dealii::update_JxW_values | dealii::update_normal_vectors;
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    IntegratorCell integrator(*matrix_free, dof_index, quad_index);

    ElementType const element_type =
      get_element_type(matrix_free->get_dof_handler(dof_index).get_triangulation());

    unsigned int const n_cells =
      matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      scalar volume      = dealii::make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = dealii::make_vectorized_array<Number>(0.0);
      if(divergence_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm or
         divergence_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(velocity);
        integrator.evaluate(dealii::EvaluationFlags::values);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          volume += integrator.JxW(q);
          norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
        }
        norm_U_mean /= volume;
      }

      scalar h     = calculate_characteristic_element_length(volume, dim, element_type);
      scalar h_eff = calculate_high_order_element_length(h, divergence_data.degree, true);

      scalar tau_convective = norm_U_mean * h_eff;
      scalar tau_viscous    = dealii::make_vectorized_array<Number>(divergence_data.viscosity);

      if(divergence_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_divergence_penalty_parameter[cell] = divergence_data.penalty_factor * tau_convective;
      }
      else if(divergence_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_divergence_penalty_parameter[cell] = divergence_data.penalty_factor * tau_viscous;
      }
      else if(divergence_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_divergence_penalty_parameter[cell] =
          divergence_data.penalty_factor * (tau_convective + tau_viscous);
      }

      if(continuity_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_continuity_penalty_parameter[cell] =
          continuity_data.penalty_factor * tau_convective / h_eff;
      }
      else if(continuity_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_continuity_penalty_parameter[cell] =
          continuity_data.penalty_factor * tau_viscous / h_eff;
      }
      else if(continuity_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_continuity_penalty_parameter[cell] =
          continuity_data.penalty_factor * (tau_convective + tau_viscous) / h_eff;
      }
    }

    velocity.zero_out_ghost_values();
  }

  void
  reinit_cell(IntegratorCell & integrator) const
  {
    tau = integrator.read_cell_data(array_divergence_penalty_parameter);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(IntegratorCell const & integrator, unsigned int const q) const
  {
    return tau * integrator.get_divergence(q);
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = 0.5 * (integrator_m.read_cell_data(array_continuity_penalty_parameter) +
                 integrator_p.read_cell_data(array_continuity_penalty_parameter));
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_continuity_penalty_parameter);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = 0.5 * (integrator_m.read_cell_data(array_continuity_penalty_parameter) +
                   integrator_p.read_cell_data(array_continuity_penalty_parameter));
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_continuity_penalty_parameter);
    }
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & u_m, vector const & u_p, vector const & normal_m) const
  {
    vector jump_value = u_m - u_p;

    vector flux;

    if(continuity_data.which_components == ContinuityPenaltyComponents::All)
    {
      // penalize all velocity components
      flux = tau * jump_value;
    }
    else if(continuity_data.which_components == ContinuityPenaltyComponents::Normal)
    {
      flux = tau * (jump_value * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    return flux;
  }

  dealii::AlignedVector<std::pair<scalar, std::array<scalar, 2 * dim>>>
  get_penalty_coefficients() const
  {
    // Collect data from faces
    std::array<VectorType, 2 * dim> accumulated_data;
    for(auto & entry : accumulated_data)
      entry.reinit(matrix_free->get_dof_handler(dof_index)
                     .get_triangulation()
                     .global_active_cell_index_partitioner()
                     .lock());

    for(unsigned int face = 0; face < matrix_free->n_inner_face_batches(); ++face)
    {
      auto const & face_info = matrix_free->get_face_info(face);
      for(unsigned int v = 0; v < matrix_free->n_active_entries_per_face_batch(face); ++v)
      {
        Number const penalty_parameter =
          0.5 * (array_continuity_penalty_parameter[face_info.cells_interior[v] / scalar::size()]
                                                   [face_info.cells_interior[v] % scalar::size()] +
                 array_continuity_penalty_parameter[face_info.cells_exterior[v] / scalar::size()]
                                                   [face_info.cells_exterior[v] % scalar::size()]);
        auto const & inner = matrix_free->get_face_iterator(face, v, true);
        accumulated_data[inner.second](inner.first->global_active_cell_index()) = penalty_parameter;
        auto const & outer = matrix_free->get_face_iterator(face, v, false);
        accumulated_data[outer.second](outer.first->global_active_cell_index()) = penalty_parameter;
      }
    }
    for(auto & entry : accumulated_data)
      entry.compress(dealii::VectorOperation::add);

    // Finally combine data accumulated from faces and fill the cell data
    dealii::AlignedVector<std::pair<scalar, std::array<scalar, 2 * dim>>> result;
    result.resize(matrix_free->n_cell_batches());
    for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
    {
      result[cell].first = array_divergence_penalty_parameter[cell];
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < scalar::size(); ++v)
          result[cell].second[face][v] = accumulated_data[face](
            matrix_free->get_cell_iterator(cell, v)->global_active_cell_index());
    }
    return result;
  }

private:
  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index;
  unsigned int quad_index;

  DivergencePenaltyKernelData divergence_data;
  ContinuityPenaltyKernelData continuity_data;

  dealii::AlignedVector<scalar> array_divergence_penalty_parameter;
  dealii::AlignedVector<scalar> array_continuity_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

template<int dim, typename Number>
class ProjectionOperator : public OperatorBase<dim, Number, dim>
{
private:
  typedef OperatorBase<dim, Number, dim> Base;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef Operators::ProjectionKernel<dim, Number> Kernel;

public:
  typedef Number value_type;

  ProjectionOperator() : velocity(nullptr), time_step_size(1.0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const &        matrix_free,
             dealii::AffineConstraints<Number> const &      affine_constraints,
             ProjectionOperatorData<dim> const &            data,
             Operators::DivergencePenaltyKernelData const & div_kernel_data,
             Operators::ContinuityPenaltyKernelData const & conti_kernel_data);

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             ProjectionOperatorData<dim> const &       data,
             std::shared_ptr<Kernel>                   kernel);

  ProjectionOperatorData<dim>
  get_data() const;

  Operators::DivergencePenaltyKernelData
  get_divergence_kernel_data() const;

  Operators::ContinuityPenaltyKernelData
  get_continuity_kernel_data() const;

  double
  get_time_step_size() const;

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  update(VectorType const & velocity, double const & dt);

  dealii::AlignedVector<std::pair<dealii::VectorizedArray<Number>,
                                  std::array<dealii::VectorizedArray<Number>, 2 * dim>>>
  get_penalty_coefficients() const
  {
    return kernel->get_penalty_coefficients();
  }


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

  ProjectionOperatorData<dim> operator_data;

  VectorType const * velocity;
  double             time_step_size;

  std::shared_ptr<Operators::ProjectionKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_PROJECTION_OPERATOR_H_ \
        */
