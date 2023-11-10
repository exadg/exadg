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

#ifndef OPERATION_BASE_H
#define OPERATION_BASE_H

// deal.II
#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/petsc_sparse_matrix.h>
#endif
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/categorization.h>
#include <exadg/matrix_free/integrators.h>

#include <exadg/solvers_and_preconditioners/preconditioners/elementwise_preconditioners.h>
#include <exadg/solvers_and_preconditioners/preconditioners/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h>
#include <exadg/solvers_and_preconditioners/utilities/invert_diagonal.h>

#include <exadg/utilities/lazy_ptr.h>

#include <exadg/operators/elementwise_operator.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/operator_type.h>

namespace ExaDG
{
struct OperatorBaseData
{
  OperatorBaseData()
    : dof_index(0),
      dof_index_inhomogeneous(dealii::numbers::invalid_unsigned_int),
      quad_index(0),
      operator_is_singular(false),
      use_cell_based_loops(false),
      implement_block_diagonal_preconditioner_matrix_free(false),
      solver_block_diagonal(Elementwise::Solver::GMRES),
      preconditioner_block_diagonal(Elementwise::Preconditioner::InverseMassMatrix),
      solver_data_block_diagonal(SolverData(1000, 1.e-12, 1.e-2, 1000))
  {
  }

  unsigned int dof_index;

  // In addition to the "standard" dof_index above, we need a separate dof index to evaluate
  // inhomogeneous boundary data correctly. This "inhomogeneous" dof index corresponds to an
  // AffineConstraints object that only applies periodicity and hanging node constraints.
  unsigned int dof_index_inhomogeneous;

  unsigned int quad_index;

  // Solution of linear systems of equations and preconditioning
  bool operator_is_singular;

  bool use_cell_based_loops;

  // block Jacobi preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // elementwise iterative solution of block Jacobi problems
  Elementwise::Solver         solver_block_diagonal;
  Elementwise::Preconditioner preconditioner_block_diagonal;
  SolverData                  solver_data_block_diagonal;
};

template<int dim, typename Number, int n_components = 1>
class OperatorBase : public dealii::Subscriptor
{
public:
  typedef OperatorBase<dim, Number, n_components> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, n_components, Number>          IntegratorCell;
  typedef FaceIntegrator<dim, n_components, Number>          IntegratorFace;

  static unsigned int const vectorization_length = dealii::VectorizedArray<Number>::size();

  typedef dealii::LAPACKFullMatrix<Number> LAPACKMatrix;

  typedef dealii::FullMatrix<dealii::TrilinosScalar> FullMatrix_;

  OperatorBase();

  virtual ~OperatorBase()
  {
  }

  /*
   *  Getters and setters.
   */
  void
  set_time(double const time) const;

  double
  get_time() const;

  unsigned int
  get_level() const;

  dealii::AffineConstraints<Number> const &
  get_affine_constraints() const;

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  /*
   * Returns whether the operator is singular, e.g., the Laplace operator with pure Neumann boundary
   * conditions is singular.
   */
  bool
  operator_is_singular() const;

  void
  vmult(VectorType & dst, VectorType const & src) const;

  void
  vmult_add(VectorType & dst, VectorType const & src) const;

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const;

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const;

  dealii::types::global_dof_index
  m() const;

  dealii::types::global_dof_index
  n() const;

  Number
  el(unsigned int const, unsigned int const) const;

  bool
  is_empty_locally() const;

  void
  initialize_dof_vector(VectorType & vector) const;

  /*
   * For time-dependent problems, the function set_time() needs to be called prior to the present
   * function.
   */
  virtual void
  set_inhomogeneous_boundary_values(VectorType & solution) const;

  void
  set_constrained_dofs_to_zero(VectorType & vector) const;

  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  /*
   * Update block diagonal preconditioner: initialize everything related to block diagonal
   * preconditioner when this function is called the first time. Recompute block matrices in case of
   * matrix-based implementation.
   */
  void
  update_block_diagonal_preconditioner() const;

  /**
   * This function applies the inverse block diagonal to a src vector and stores the result in the
   * dst vector. This function may be called with identical dst, src vectors.
   */
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

  /*
   * Algebraic multigrid (AMG): sparse matrix (Trilinos) methods
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix,
                     MPI_Comm const &                         mpi_comm) const;

  void
  calculate_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const;
#endif

  /*
   * Algebraic multigrid (AMG): sparse matrix (PETSc) methods
   */
#ifdef DEAL_II_WITH_PETSC
  void
  init_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
                     MPI_Comm const &                           mpi_comm) const;

  void
  calculate_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const;
#endif

  /*
   * Evaluate the homogeneous part of an operator. The homogeneous operator is the operator that is
   * obtained for homogeneous boundary conditions. This operation is typically applied in linear
   * iterative solvers (as well as multigrid preconditioners and smoothers). Operations of this type
   * are called apply_...() and vmult_...() as required by deal.II interfaces.
   */
  void
  apply(VectorType & dst, VectorType const & src) const;

  /*
   * See function apply() for a description.
   */
  void
  apply_add(VectorType & dst, VectorType const & src) const;

  /*
   * evaluate inhomogeneous parts of operator related to inhomogeneous boundary face integrals.
   * Operations of this type are called rhs_...() since these functions are called to calculate the
   * vector forming the right-hand side vector of linear systems of equations. Functions of type
   * rhs only make sense for linear operators (but they have e.g. no meaning for linearized
   * operators of nonlinear problems). For this reason, these functions are currently defined
   * 'virtual' to provide the opportunity to override and assert these functions in derived classes.
   *
   * For continuous Galerkin discretizations, this function calls internally the member function
   * set_inhomogeneous_boundary_values(). Hence, prior to calling this function, one needs to call
   * set_time() for a correct evaluation in case of time-dependent problems.
   *
   * This function sets the dst vector to zero, and afterwards calls rhs_add().
   */
  virtual void
  rhs(VectorType & dst) const;

  /*
   * See function rhs() for a description.
   */
  virtual void
  rhs_add(VectorType & dst) const;

  /*
   * Evaluate the operator including homogeneous and inhomogeneous contributions. The typical use
   * case would be explicit time integration where a splitting into homogeneous and inhomogeneous
   * contributions is not required. Functions of type evaluate only make sense for linear operators
   * (but they have e.g. no meaning for linearized operators of nonlinear problems). For this
   * reason, these functions are currently defined 'virtual' to provide the opportunity to override
   * and assert these functions in derived classes.
   *
   * Unlike the function rhs(), this function does not internally call the function
   * set_inhomogeneous_boundary_values() prior to evaluation. Hence, one needs to explicitly call
   * the function set_inhomogeneous_boundary_values() in case of continuous Galerkin discretizations
   * with inhomogeneous Dirichlet boundary conditions before calling the present function.
   */
  virtual void
  evaluate(VectorType & dst, VectorType const & src) const;

  /*
   * See function evaluate() for a description.
   */
  virtual void
  evaluate_add(VectorType & dst, VectorType const & src) const;

  /*
   * point Jacobi preconditioner (diagonal)
   */
  void
  calculate_diagonal(VectorType & diagonal) const;

  void
  add_diagonal(VectorType & diagonal) const;

  /*
   * block Jacobi preconditioner (block-diagonal)
   */

  // matrix-based implementation
  void
  calculate_block_diagonal_matrices() const;

  void
  add_block_diagonal_matrices(std::vector<LAPACKMatrix> & matrices) const;

  void
  apply_block_diagonal_matrix_based(VectorType & dst, VectorType const & src) const;

  void
  apply_inverse_block_diagonal_matrix_based(VectorType & dst, VectorType const & src) const;

  // matrix-free implementation

  // This function has to initialize everything related to the block diagonal preconditioner when
  // using the matrix-free variant with elementwise iterative solvers and matrix-free operator
  // evaluation.
  void
  initialize_block_diagonal_preconditioner_matrix_free() const;

  void
  apply_add_block_diagonal_elementwise(unsigned int const                            cell,
                                       dealii::VectorizedArray<Number> * const       dst,
                                       dealii::VectorizedArray<Number> const * const src,
                                       unsigned int const problem_size) const;

  /*
   * additive Schwarz preconditioner (cellwise block-diagonal)
   */
  virtual void
  compute_factorized_additive_schwarz_matrices() const;

  void
  apply_inverse_additive_schwarz_matrices(VectorType & dst, VectorType const & src) const;

protected:
  void
  reinit(dealii::MatrixFree<dim, Number> const &   matrix_free,
         dealii::AffineConstraints<Number> const & constraints,
         OperatorBaseData const &                  data);

  void
  reinit_cell(IntegratorCell & integrator, unsigned int const cell) const;

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const face) const;

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const face) const;

  void
  reinit_face_cell_based(IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const;

  /*
   * These methods have to be overwritten by derived classes because these functions are
   * operator-specific and define how the operator looks like.
   */

  // standard integration procedure with separate loops for cell and face integrals
  virtual void
  do_cell_integral(IntegratorCell & integrator) const;

  virtual void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  virtual void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  virtual void
  do_boundary_integral_continuous(IntegratorFace &                   integrator,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const;

  // The computation of the diagonal and block-diagonal requires face integrals of type
  // interior (int) and exterior (ext)
  virtual void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  virtual void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  // Since only special operators need to evaluate neighboring data, this function
  // simply redirects to do_face_int_integral() if this function is not overwritten
  // by a derived class (such as convective terms that require an additional
  // evaluation of velocity fields for example).
  virtual void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const;

  /*
   * Matrix-free object.
   */
  lazy_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  /*
   * Physical time (required for time-dependent problems).
   */
  mutable double time;

  /*
   * Constraint matrix.
   */
  lazy_ptr<dealii::AffineConstraints<Number>> constraint;

  mutable dealii::AffineConstraints<double> constraint_double;

  /*
   * Cell and face integrator flags.
   */
  IntegratorFlags integrator_flags;

  /*
   * Is the operator used as a multigrid level operator?
   */
  bool is_mg;

  /*
   * Is the discretization based on discontinuous Galerkin method?
   */
  bool is_dg;

  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef Elementwise::OperatorBase<dim, Number, This> ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<dealii::VectorizedArray<Number>>
    ELEMENTWISE_PRECONDITIONER;
  typedef Elementwise::
    IterativeSolver<dim, n_components, Number, ELEMENTWISE_OPERATOR, ELEMENTWISE_PRECONDITIONER>
      ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR>       elementwise_operator;
  mutable std::shared_ptr<ELEMENTWISE_PRECONDITIONER> elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>         elementwise_solver;

private:
  virtual void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const;

  virtual void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const;

  virtual void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const;

  virtual void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const;

  /*
   * Helper functions:
   *
   * The diagonal, block-diagonal, as well as the system matrix (assembled into a sparse matrix) are
   * computed columnwise. This means that column i of the block-matrix is computed by evaluating the
   * operator for a unit vector which takes a value of 1 in row i and is 0 for all other entries.
   */
  void
  create_standard_basis(unsigned int j, IntegratorCell & integrator) const;

  void
  create_standard_basis(unsigned int j, IntegratorFace & integrator) const;

  void
  create_standard_basis(unsigned int     j,
                        IntegratorFace & integrator_1,
                        IntegratorFace & integrator_2) const;

  /*
   * This function calculates cell integrals for the full (= homogeneous + inhomogeneous) part,
   * where inhomogeneous part may occur for continuous Galerkin discretizations due to inhomogeneous
   * Dirichlet BCs. A prerequisite to call this function is to set inhomogeneous Dirichlet degrees
   * of freedom appropriately in the DoF-vector src. In case the DoF-vector src is zero apart from
   * the inhomogeneous Dirichlet degrees of freedom, this function calculates only the inhomogeneous
   * part of the operator, because the homogeneous operator is zero in this case. For DG
   * discretizations, this function is equivalent to cell_loop().
   */
  void
  cell_loop_full_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                          VectorType &                            dst,
                          VectorType const &                      src,
                          Range const &                           range) const;

  /*
   * This function loops over all cells and calculates cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           range) const;

  /*
   * This function loops over all interior faces and calculates face integrals.
   */
  void
  face_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           range) const;

  /*
   * The following functions loop over all boundary faces and calculate boundary face integrals.
   * Depending on the operator type, we distinguish between boundary face integrals of type
   * homogeneous, inhomogeneous, and full.
   */

  // homogeneous operator
  void
  boundary_face_loop_hom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                            dst,
                                  VectorType const &                      src,
                                  Range const &                           range) const;

  // inhomogeneous operator
  void
  boundary_face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           range) const;

  // full operator
  void
  boundary_face_loop_full_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                            dst,
                                   VectorType const &                      src,
                                   Range const &                           range) const;

  /*
   * inhomogeneous operator: For the inhomogeneous operator, we only have to calculate boundary face
   * integrals. The matrix-free implementation, however, does not offer interfaces for boundary face
   * integrals only. Hence we have to provide empty functions for cell and interior face integrals.
   */
  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  /*
   * Calculate diagonal.
   */
  void
  cell_loop_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                            dst,
                     VectorType const &                      src,
                     Range const &                           range) const;

  void
  face_loop_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                            dst,
                     VectorType const &                      src,
                     Range const &                           range) const;

  void
  boundary_face_loop_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                              VectorType &                            dst,
                              VectorType const &                      src,
                              Range const &                           range) const;

  void
  cell_based_loop_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           range) const;

  /*
   * Calculate (assemble) block diagonal.
   */
  void
  cell_loop_block_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                           std::vector<LAPACKMatrix> &             matrices,
                           std::vector<LAPACKMatrix> const &       src,
                           Range const &                           range) const;

  void
  face_loop_block_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                           std::vector<LAPACKMatrix> &             matrices,
                           std::vector<LAPACKMatrix> const &       src,
                           Range const &                           range) const;

  void
  boundary_face_loop_block_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    std::vector<LAPACKMatrix> &             matrices,
                                    std::vector<LAPACKMatrix> const &       src,
                                    Range const &                           range) const;

  // cell-based variant for computation of both cell and face integrals
  void
  cell_based_loop_block_diagonal(dealii::MatrixFree<dim, Number> const & matrix_free,
                                 std::vector<LAPACKMatrix> &             matrices,
                                 std::vector<LAPACKMatrix> const &       src,
                                 Range const &                           range) const;

  /*
   * Apply block diagonal.
   */
  void
  cell_loop_apply_block_diagonal_matrix_based(dealii::MatrixFree<dim, Number> const & matrix_free,
                                              VectorType &                            dst,
                                              VectorType const &                      src,
                                              Range const &                           range) const;

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal_matrix_based(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           range) const;

  /*
   * Set up sparse matrix internally for templated matrix type (Trilinos or
   * PETSc matrices)
   */
  template<typename SparseMatrix>
  void
  internal_init_system_matrix(SparseMatrix &                   system_matrix,
                              dealii::DynamicSparsityPattern & dsp,
                              MPI_Comm const &                 mpi_comm) const;

  template<typename SparseMatrix>
  void
  internal_calculate_system_matrix(SparseMatrix & system_matrix) const;

  /*
   * Calculate sparse matrix.
   */
  template<typename SparseMatrix>
  void
  cell_loop_calculate_system_matrix(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    SparseMatrix &                          dst,
                                    SparseMatrix const &                    src,
                                    Range const &                           range) const;

  template<typename SparseMatrix>
  void
  face_loop_calculate_system_matrix(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    SparseMatrix &                          dst,
                                    SparseMatrix const &                    src,
                                    Range const &                           range) const;

  template<typename SparseMatrix>
  void
  boundary_face_loop_calculate_system_matrix(dealii::MatrixFree<dim, Number> const & matrix_free,
                                             SparseMatrix &                          dst,
                                             SparseMatrix const &                    src,
                                             Range const &                           range) const;

  /*
   * This function sets entries of the DoF-vector corresponding to constraint DoFs to one.
   */
  void
  set_constrained_dofs_to_one(VectorType & vector) const;

  /*
   * Do we have to evaluate (boundary) face integrals for this operator? For example, operators
   * such as the mass operator only involve cell integrals.
   */
  bool
  evaluate_face_integrals() const;

  /*
   * Compute factorized additive Schwarz matrices.
   */
  template<typename SparseMatrix>
  void
  internal_compute_factorized_additive_schwarz_matrices() const;

  /*
   * Data structure containing all operator-specific data.
   */
  OperatorBaseData data;

  /*
   * Multigrid level: 0 <= level <= max_level. If the operator is not used as a multigrid
   * level operator, this variable takes a value of dealii::numbers::invalid_unsigned_int.
   */
  unsigned int level;

  /*
   * Vector of matrices for block-diagonal preconditioners.
   */
  mutable std::vector<LAPACKMatrix> matrices;

  /*
   * Vector with weights for additive Schwarz preconditioner.
   */
  mutable VectorType weights;

  /*
   * We want to initialize the block diagonal preconditioner (block diagonal matrices or elementwise
   * iterative solvers in case of matrix-free implementation) only once, so we store the status of
   * initialization in a variable.
   */
  mutable bool block_diagonal_preconditioner_is_initialized;

  unsigned int n_mpi_processes;
};
} // namespace ExaDG

#endif
