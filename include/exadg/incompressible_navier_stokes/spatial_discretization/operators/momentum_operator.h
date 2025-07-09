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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/generalized_newtonian_model.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/turbulence_model.h>
#include <exadg/incompressible_navier_stokes/user_interface/viscosity_model_data.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct MomentumOperatorData : public OperatorBaseData
{
  MomentumOperatorData()
    : OperatorBaseData(), unsteady_problem(false), convective_problem(false), viscous_problem(false)
  {
  }

  bool unsteady_problem;
  bool convective_problem;
  bool viscous_problem;

  TurbulenceModelData           turbulence_model_data;
  GeneralizedNewtonianModelData generalized_newtonian_model_data;

  Operators::ConvectiveKernelData convective_kernel_data;
  Operators::ViscousKernelData    viscous_kernel_data;

  std::shared_ptr<BoundaryDescriptorU<dim> const> bc;
};

template<int dim, typename Number>
class MomentumOperator : public OperatorBase<dim, Number, dim>
{
private:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef OperatorBase<dim, Number, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

public:
  // required by preconditioner interfaces
  typedef Number value_type;

  MomentumOperator();

  /**
   * This function creates own kernels for the different terms of the combined PDE operator. This
   * function is typically called when using this operator as the PDE operator in multigrid.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MomentumOperatorData<dim> const &         data,
             dealii::Mapping<dim> const &              mapping);

  void
  initialize(dealii::MatrixFree<dim, Number> const &                   matrix_free,
             dealii::AffineConstraints<Number> const &                 affine_constraints,
             MomentumOperatorData<dim> const &                         data,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel,
             std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel);

  MomentumOperatorData<dim> const &
  get_data() const;

  Operators::ConvectiveKernelData
  get_convective_kernel_data() const;

  Operators::ViscousKernelData
  get_viscous_kernel_data() const;

  VectorType const &
  get_velocity() const;

  /*
   * Interface required by Newton solver.
   */
  void
  set_solution_linearization(VectorType const & velocity);

  /*
   * Update of operator (required, e.g., for multigrid).
   */
  void
  update_after_grid_motion();

  void
  set_velocity_copy(VectorType const & velocity) const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

  Number
  get_scaling_factor_mass_operator() const;

  void
  set_scaling_factor_mass_operator(Number const & number);

  void
  update_viscosity(VectorType const & velocity) const;

  /*
   * Interfaces of OperatorBase.
   */
  void
  rhs(VectorType & dst) const final;

  void
  rhs_add(VectorType & dst) const final;

  void
  evaluate(VectorType & dst, VectorType const & src) const final;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const final;

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

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const final;

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  MomentumOperatorData<dim> operator_data;

  std::shared_ptr<MassKernel<dim, Number>>                  mass_kernel;
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel;

  // Flag signaling that the `viscous_kernel` is managed by this class.
  bool viscous_kernel_own_storage;

  // Variable viscosity models for when `viscous_kernel` is managed by this class.
  mutable TurbulenceModel<dim, Number>           turbulence_model_own_storage;
  mutable GeneralizedNewtonianModel<dim, Number> generalized_newtonian_model_own_storage;

  double scaling_factor_mass;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_ \
        */
