/*
 * velocity_conv_diff_operator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include "operators/convective_operator.h"
#include "operators/mass_matrix_operator.h"
#include "operators/viscous_operator.h"

#include "../../operators/elementwise_operator.h"
#include "../../operators/linear_operator_base.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

#include "solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

namespace IncNS
{
template<int dim>
struct MomentumOperatorData
{
  MomentumOperatorData()
    : unsteady_problem(true),
      convective_problem(true),
      dof_index(0),
      quad_index_std(0),
      quad_index_over(1),
      scaling_factor_time_derivative_term(-1.0),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-1 /*rel_tol TODO*/, 1000)),
      mg_operator_type(MultigridOperatorType::Undefined)
  {
  }

  bool unsteady_problem;
  bool convective_problem;

  unsigned int dof_index;

  unsigned int quad_index_std;
  unsigned int quad_index_over;

  double scaling_factor_time_derivative_term;

  MassMatrixOperatorData      mass_matrix_operator_data;
  ViscousOperatorData<dim>    viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;

  // Multigrid
  MultigridOperatorType mg_operator_type;
};

template<int dim, typename Number = double>
class MomentumOperator : public LinearOperatorBase
{
public:
  typedef MomentumOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  static const int DIM = dim;
  typedef Number   value_type;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  MomentumOperator();

  virtual ~MomentumOperator()
  {
  }

  void
  reinit_multigrid(MatrixFree<dim, Number> const &   data,
                   AffineConstraints<double> const & constraint_matrix,
                   MomentumOperatorData<dim> const & operator_data);

  void
  reinit(MatrixFree<dim, Number> const &         data,
         MomentumOperatorData<dim> const &       operator_data,
         MassMatrixOperator<dim, Number> const & mass_matrix_operator,
         ViscousOperator<dim, Number> const &    viscous_operator,
         ConvectiveOperator<dim, Number> const & convective_operator);

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const
  {
    vmult(dst, src);
  }

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const
  {
    vmult_add(dst, src);
  }

  types::global_dof_index
  m() const
  {
    return n();
  }

  types::global_dof_index
  n() const
  {
    MatrixFree<dim, Number> const & data      = get_data();
    unsigned int                    dof_index = get_dof_index();

    return data.get_vector_partitioner(dof_index)->size();
  }

  Number
  el(const unsigned int, const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  bool
  is_empty_locally() const
  {
    MatrixFree<dim, Number> const & data = get_data();
    return (data.n_macro_cells() == 0);
  }

  void
  initialize_dof_vector(VectorType & vector) const
  {
    MatrixFree<dim, Number> const & data      = get_data();
    unsigned int                    dof_index = get_dof_index();

    data.initialize_dof_vector(vector, dof_index);
  }

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const
  {
    AssertThrow(false,
                ExcMessage("MomentumOperator::get_constraint_matrix should be overwritten!"));
    return *(new AffineConstraints<double>());
  }

  virtual bool
  is_singular() const
  {
    // per default the operator is not singular
    // if an operator can be singular, this method has to be overwritten
    return false;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false, ExcMessage("MomentumOperator::init_system_matrix should be overwritten!"));
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false,
                ExcMessage("MomentumOperator::calculate_system_matrix should be overwritten!"));
  }
#endif

  /*
   * Setters and getters.
   */

  MatrixFree<dim, Number> const &
  get_data() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void
  set_scaling_factor_time_derivative_term(double const & factor);

  double
  get_scaling_factor_time_derivative_term() const;

  /*
   *  Linearized velocity field for convective operator
   */
  void
  set_solution_linearization(VectorType const & solution_linearization);

  VectorType const &
  get_solution_linearization() const;

  /*
   *  Evaluation time that is needed for evaluation of linearized convective operator.
   */
  void
  set_evaluation_time(double const time);

  double
  get_evaluation_time() const;

  /*
   *  Operator data
   */
  MomentumOperatorData<dim> const &
  get_operator_data() const;

  /*
   *  Operator data of basic operators: mass matrix, convective operator, viscous operator
   */
  MassMatrixOperatorData const &
  get_mass_matrix_operator_data() const;

  ConvectiveOperatorData<dim> const &
  get_convective_operator_data() const;

  ViscousOperatorData<dim> const &
  get_viscous_operator_data() const;

  /*
   *  This function applies the matrix vector multiplication.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  /*
   *  This function applies matrix vector product and adds the result
   *  to the dst-vector.
   */
  void
  vmult_add(VectorType & dst, VectorType const & src) const;


  /*
   *  This function applies the matrix-vector multiplication for the block Jacobi operation.
   */
  void
  vmult_block_jacobi(VectorType & dst, VectorType const & src) const;

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  /*
   *  Apply block Jacobi preconditioner.
   */
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;


  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void
  update_block_diagonal_preconditioner() const;

  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src,
                                       unsigned int const problem_size = 1) const;

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void
  calculate_diagonal(VectorType & diagonal) const;

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<value_type>> & matrices) const;

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  initialize_block_diagonal_preconditioner_matrix_free() const;

  /*
   * Verify computation of block Jacobi matrices.
   */
  void
  check_block_jacobi_matrices() const;

  /*
   *  This function is only needed for testing.
   */
  void
  cell_loop_apply_block_diagonal(MatrixFree<dim, Number> const &               data,
                                 VectorType &                                  dst,
                                 VectorType const &                            src,
                                 std::pair<unsigned int, unsigned int> const & cell_range) const;

  MomentumOperatorData<dim> operator_data;

  MatrixFree<dim, Number> const * data;

  MassMatrixOperator<dim, Number> const * mass_matrix_operator;

  ViscousOperator<dim, Number> const * viscous_operator;

  ConvectiveOperator<dim, Number> const * convective_operator;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the velocity convection-diffusion operator.
   * In that case, the VelocityConvDiffOperator has to be generated
   * for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MassMatrixOperator, ViscousOperator,
   *   e.g., own_mass_matrix_operator_storage.reinit(...);
   * and later initialize the VelocityConvDiffOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_mass_matrix_operator_storage;
   */
  MassMatrixOperator<dim, Number> own_mass_matrix_operator_storage;

  ViscousOperator<dim, Number> own_viscous_operator_storage;

  ConvectiveOperator<dim, Number> own_convective_operator_storage;

  VectorType mutable temp_vector;
  VectorType mutable velocity_linearization;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

  /*
   * Vector of matrices for block-diagonal preconditioners.
   */
  mutable std::vector<LAPACKFullMatrix<Number>> matrices;

  /*
   * We want to initialize the block diagonal preconditioner (block diagonal matrices or elementwise
   * iterative solvers in case of matrix-free implementation) only once, so we store the status of
   * initialization in a variable.
   */
  mutable bool block_diagonal_preconditioner_is_initialized;

  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef Elementwise::OperatorBase<dim, Number, This>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PRECONDITIONER_BASE;
  typedef Elementwise::IterativeSolver<dim, dim, Number, ELEMENTWISE_OPERATOR, PRECONDITIONER_BASE>
    ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR> elementwise_operator;
  mutable std::shared_ptr<PRECONDITIONER_BASE>  elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>   elementwise_solver;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_ \
        */
