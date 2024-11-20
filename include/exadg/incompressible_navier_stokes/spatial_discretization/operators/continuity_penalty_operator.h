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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_

#include <exadg/grid/calculate_characteristic_element_length.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/operator_type.h>

namespace ExaDG
{
namespace IncNS
{
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

template<int dim, typename Number>
class ContinuityPenaltyKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  ContinuityPenaltyKernel()
    : matrix_free(nullptr), dof_index(0), quad_index(0), array_penalty_parameter(0)
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index,
         ContinuityPenaltyKernelData const &     data)
  {
    this->matrix_free = &matrix_free;

    this->dof_index  = dof_index;
    this->quad_index = quad_index;

    this->data = data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);
  }

  ContinuityPenaltyKernelData
  get_data()
  {
    return this->data;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    // no cell integrals

    flags.face_evaluate  = dealii::EvaluationFlags::values;
    flags.face_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    // no cell integrals

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

    dealii::AlignedVector<scalar> JxW_values(integrator.n_q_points);

    ElementType const element_type =
      get_element_type(matrix_free->get_dof_handler(dof_index).get_triangulation());

    unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(velocity);
      integrator.evaluate(dealii::EvaluationFlags::values);
      scalar volume      = dealii::make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        volume += integrator.JxW(q);
        norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
      }

      norm_U_mean /= volume;

      scalar tau_convective = norm_U_mean;
      scalar h              = calculate_characteristic_element_length(volume, dim, element_type);
      scalar h_eff          = calculate_high_order_element_length(h, data.degree, true);
      scalar tau_viscous    = dealii::make_vectorized_array<Number>(data.viscosity) / h_eff;

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_convective;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_viscous;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] = data.penalty_factor * (tau_convective + tau_viscous);
      }
    }
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                 integrator_p.read_cell_data(array_penalty_parameter));
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                   integrator_p.read_cell_data(array_penalty_parameter));
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter);
    }
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & u_m, vector const & u_p, vector const & normal_m) const
  {
    vector jump_value = u_m - u_p;

    vector flux;

    if(data.which_components == ContinuityPenaltyComponents::All)
    {
      // penalize all velocity components
      flux = tau * jump_value;
    }
    else if(data.which_components == ContinuityPenaltyComponents::Normal)
    {
      flux = tau * (jump_value * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    return flux;
  }


private:
  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index;
  unsigned int quad_index;

  ContinuityPenaltyKernelData data;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

template<int dim>
struct ContinuityPenaltyData
{
  ContinuityPenaltyData() : dof_index(0), quad_index(0), use_boundary_data(false)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;

  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};

template<int dim, typename Number>
class ContinuityPenaltyOperator
{
private:
  typedef ContinuityPenaltyOperator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef Operators::ContinuityPenaltyKernel<dim, Number> Kernel;

public:
  ContinuityPenaltyOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             ContinuityPenaltyData<dim> const &      data,
             std::shared_ptr<Kernel> const           kernel);

  void
  update(VectorType const & velocity);

  // homogeneous operator
  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

  // inhomogeneous operator
  void
  rhs(VectorType & dst, Number const evaluation_time) const;

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const;

  // full operator, i.e., homogeneous and inhomogeneous contributions
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const;

private:
  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  face_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           face_range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           face_range) const;

  void
  boundary_face_loop_hom(dealii::MatrixFree<dim, Number> const & matrix_free,
                         VectorType &                            dst,
                         VectorType const &                      src,
                         Range const &                           face_range) const;

  void
  boundary_face_loop_full(dealii::MatrixFree<dim, Number> const & matrix_free,
                          VectorType &                            dst,
                          VectorType const &                      src,
                          Range const &                           face_range) const;

  void
  boundary_face_loop_inhom(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           face_range) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  ContinuityPenaltyData<dim> data;

  mutable double time;

  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_ \
        */
