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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/continuity_penalty_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/divergence_penalty_operator.h>
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
      use_boundary_data(false)
  {
  }

  // specify which penalty terms are used
  bool use_divergence_penalty, use_continuity_penalty;

  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};

template<int dim, typename Number>
class ProjectionOperator : public OperatorBase<dim, Number, dim>
{
private:
  typedef OperatorBase<dim, Number, dim> Base;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef Operators::DivergencePenaltyKernel<dim, Number> DivKernel;
  typedef Operators::ContinuityPenaltyKernel<dim, Number> ContiKernel;

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
             std::shared_ptr<DivKernel>                div_penalty_kernel,
             std::shared_ptr<ContiKernel>              conti_penalty_kernel);

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

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_kernel;
  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_PROJECTION_OPERATOR_H_ \
        */
