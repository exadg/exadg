/*
 * momentum_operator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h>
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

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MomentumOperatorData<dim> const &         data);

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

  dealii::LinearAlgebra::distributed::Vector<Number> const &
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
  reinit_cell_additional(IntegratorCell & integrator, unsigned int const cell) const final;

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const face) const final;

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const face) const final;

  void
  reinit_face_cell_based(IntegratorFace &                 integrator_m,
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

  double scaling_factor_mass;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MOMENTUM_OPERATOR_H_ \
        */
