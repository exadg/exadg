/*
 * momentum_operator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_

#include "operators/convective_operator.h"
#include "operators/mass_matrix_operator.h"
#include "operators/viscous_operator.h"

#include "../../operators/operator_base.h"

namespace IncNS
{
template<int dim>
struct MomentumOperatorData : public OperatorBaseData
{
  MomentumOperatorData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      unsteady_problem(false),
      convective_problem(false),
      viscous_problem(false),
      scaling_factor_mass_matrix(1.0),
      mg_operator_type(MultigridOperatorType::Undefined)
  {
  }

  bool unsteady_problem;
  bool convective_problem;
  bool viscous_problem;

  // This variable is only needed for initialization, e.g., so that the
  // multigrid smoothers can be initialized correctly during initialization
  // (e.g., diagonal in case of Chebyshev smoother) without the need to
  // update the whole multigrid preconditioner in the first time step.
  double scaling_factor_mass_matrix;

  Operators::ConvectiveKernelData convective_kernel_data;
  Operators::ViscousKernelData    viscous_kernel_data;

  // Multigrid
  MultigridOperatorType mg_operator_type;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, typename Number>
class MomentumOperator : public OperatorBase<dim, Number, MomentumOperatorData<dim>, dim>
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef OperatorBase<dim, Number, MomentumOperatorData<dim>, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

public:
  // required by preconditioner interfaces
  typedef Number value_type;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         MomentumOperatorData<dim> const & operator_data) const;

  void
  reinit(MatrixFree<dim, Number> const &                           matrix_free,
         AffineConstraints<double> const &                         constraint_matrix,
         MomentumOperatorData<dim> const &                         operator_data,
         std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel,
         std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel);

  LinearAlgebra::distributed::Vector<Number> const &
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
  set_velocity_copy(VectorType const & velocity) const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

  Number
  get_scaling_factor_mass_matrix() const;

  void
  set_scaling_factor_mass_matrix(Number const & number) const;

  /*
   * Interfaces of OperatorBase.
   */
  void
  rhs(VectorType & dst) const;

  void
  rhs_add(VectorType & dst) const;

  void
  evaluate(VectorType & dst, VectorType const & src) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const;

  void
  reinit_cell(unsigned int const cell) const;

  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const;

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

private:
  mutable std::shared_ptr<Operators::MassMatrixKernel<dim, Number>> mass_kernel;
  mutable std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  mutable std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel;
};



} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_ \
        */
