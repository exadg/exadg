/*
 * convection_diffusion_operator_merged.h
 *
 *  Created on: Jun 6, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_MERGED_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_MERGED_H_

#include "convective_operator.h"
#include "diffusive_operator.h"
#include "mass_operator.h"

#include "../../../operators/elementwise_operator.h"
#include "../../../solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "../../../solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

namespace ConvDiff
{
template<int dim>
struct ConvectionDiffusionOperatorMergedData : public OperatorBaseData
{
  ConvectionDiffusionOperatorMergedData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      unsteady_problem(false),
      convective_problem(false),
      diffusive_problem(false),
      scaling_factor_mass_matrix(1.0),
      // TODO
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2, 1000)),
      mg_operator_type(MultigridOperatorType::Undefined)
  {
  }

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;

  // This variable is only needed for initialization, e.g., so that the
  // multigrid smoothers can be initialized correctly during initialization
  // (e.g., diagonal in case of Chebyshev smoother) without the need to
  // update the whole multigrid preconditioner in the first time step.
  double scaling_factor_mass_matrix;

  Operators::ConvectiveKernelData<dim> convective_kernel_data;
  Operators::DiffusiveKernelData       diffusive_kernel_data;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;

  // TODO
  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;

  MultigridOperatorType mg_operator_type;
};

template<int dim, typename Number>
class ConvectionDiffusionOperatorMerged
  : public OperatorBase<dim, Number, ConvectionDiffusionOperatorMergedData<dim>>
{
public:
  typedef Number value_type;

private:
  typedef OperatorBase<dim, Number, ConvectionDiffusionOperatorMergedData<dim>> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  void
  reinit(MatrixFree<dim, Number> const &                    matrix_free,
         AffineConstraints<double> const &                  constraint_matrix,
         ConvectionDiffusionOperatorMergedData<dim> const & operator_data) const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  set_velocity_copy(VectorType const & velocity) const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

  Number
  get_scaling_factor_mass_matrix() const;

  void
  set_scaling_factor_mass_matrix(Number const & number) const;

  // TODO shift to base class if possible
  // The base operator only implements the matrix-based variant of this operation.
  // Hence, this function has to be overwritten here to support both matrix-based
  // and matrix-free variants of the inverse block diagonal operator.
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

private:
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

  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &           integrator_m,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  // TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
  // cell-based face loops
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const;

  void
  do_verify_boundary_conditions(types::boundary_id const                           boundary_id,
                                ConvectionDiffusionOperatorMergedData<dim> const & operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  // TODO shift to base class if possible
  // This function has to initialize everything related to the block diagonal preconditioner when
  // using the matrix-free variant with elementwise iterative solvers and matrix-free operator
  // evaluation.
  void
  initialize_block_diagonal_preconditioner_matrix_free() const;

  Operators::MassMatrixKernel<dim, Number> mass_kernel;
  Operators::ConvectiveKernel<dim, Number> convective_kernel;
  Operators::DiffusiveKernel<dim, Number>  diffusive_kernel;

  // TODO shift to base class if possible
  // Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
  typedef Elementwise::OperatorBase<dim, Number, Base>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> ELEMENTWISE_PRECONDITIONER;
  typedef Elementwise::IterativeSolver<dim,
                                       1 /*scalar equation*/,
                                       Number,
                                       ELEMENTWISE_OPERATOR,
                                       ELEMENTWISE_PRECONDITIONER>
    ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR>       elementwise_operator;
  mutable std::shared_ptr<ELEMENTWISE_PRECONDITIONER> elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>         elementwise_solver;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_MERGED_H_ \
        */
