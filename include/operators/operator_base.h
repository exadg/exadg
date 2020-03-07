#ifndef OPERATION_BASE_H
#define OPERATION_BASE_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "operator_type.h"

#include "../functionalities/lazy_ptr.h"

#include "../solvers_and_preconditioners/util/invert_diagonal.h"

#include "integrator_flags.h"
#include "mapping_flags.h"

#include "../solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "../solvers_and_preconditioners/preconditioner/enum_types.h"
#include "../solvers_and_preconditioners/solvers/enum_types.h"
#include "../solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"
#include "elementwise_operator.h"

using namespace dealii;

struct OperatorBaseData
{
  OperatorBaseData(const unsigned int dof_index, const unsigned int quad_index)
    : dof_index(dof_index),
      quad_index(quad_index),
      operator_is_singular(false),
      use_cell_based_loops(false),
      implement_block_diagonal_preconditioner_matrix_free(false),
      solver_block_diagonal(Elementwise::Solver::GMRES),
      preconditioner_block_diagonal(Elementwise::Preconditioner::InverseMassMatrix),
      solver_data_block_diagonal(SolverData(1000, 1.e-12, 1.e-2, 1000))
  {
  }

  unsigned int dof_index;
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

template<int dim, typename Number, typename AdditionalData, int n_components = 1>
class OperatorBase : public dealii::Subscriptor
{
public:
  typedef OperatorBase<dim, Number, AdditionalData, n_components> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>      Range;
  typedef CellIntegrator<dim, n_components, Number>  IntegratorCell;
  typedef FaceIntegrator<dim, n_components, Number>  IntegratorFace;

  /*
   * Solution of linear systems of equations and preconditioning
   */
  static const unsigned int vectorization_length = VectorizedArray<Number>::n_array_elements;

  typedef std::vector<LAPACKFullMatrix<Number>> BlockMatrix;

  typedef typename GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>
    PeriodicFacePairIterator;

#ifdef DEAL_II_WITH_TRILINOS
  typedef FullMatrix<TrilinosScalar>     FullMatrix_;
  typedef TrilinosWrappers::SparseMatrix SparseMatrix;
#endif

  OperatorBase();

  virtual ~OperatorBase()
  {
  }

  virtual void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         AdditionalData const &            operator_data);

  /*
   *  Getters and setters.
   */
  AdditionalData const &
  get_data() const;

  void
  set_time(double const time) const;

  double
  get_time() const;

  unsigned int
  get_level() const;

  AffineConstraints<double> const &
  get_constraint_matrix() const;

  MatrixFree<dim, Number> const &
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

  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  Number
  el(const unsigned int, const unsigned int) const;

  bool
  is_empty_locally() const;

  void
  initialize_dof_vector(VectorType & vector) const;

  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  /*
   * Update block diagonal preconditioner: initialize everything related to block diagonal
   * preconditioner when this function is called the first time. Recompute block matrices in case of
   * matrix-based implementation.
   */
  void
  update_block_diagonal_preconditioner() const;

  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

  /*
   * Algebraic multigrid (AMG): sparse matrix (Trilinos) methods
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(SparseMatrix & system_matrix) const;

  void
  calculate_system_matrix(SparseMatrix & system_matrix) const;
#endif

  /*
   * Evaluate the homogeneous part of an operator. The homogeneous operator is the operator that is
   * obtained for homogeneous boundary conditions. This operation is typically applied in linear
   * iterative solvers (as well as multigrid preconditioners and smoothers). Operations of this type
   * are called apply_...() and vmult_...() as required by deal.II interfaces.
   */
  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

  /*
   * evaluate inhomogeneous parts of operator related to inhomogeneous boundary face integrals.
   * Operations of this type are called rhs_...() since these functions are called to calculate the
   * vector forming the right-hand side vector of linear systems of equations. Functions of type
   * rhs only make sense for linear operators (but they have e.g. no meaning for linearized
   * operators of nonlinear problems). For this reason, these functions are currently defined
   * 'virtual' to provide the opportunity to override and assert these functions in derived classes.
   */
  virtual void
  rhs(VectorType & dst) const;

  virtual void
  rhs_add(VectorType & dst) const;

  /*
   * Evaluate the operator including homogeneous and inhomogeneous contributions. The typical use
   * case would be explicit time integration where a splitting into homogeneous and inhomogeneous
   * contributions is not required. Functions of type evaluate only make sense for linear operators
   * (but they have e.g. no meaning for linearized operators of nonlinear problems). For this
   * reason, these functions are currently defined 'virtual' to provide the opportunity to override
   * and assert these functions in derived classes.
   */
  virtual void
  evaluate(VectorType & dst, VectorType const & src) const;

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
  add_block_diagonal_matrices(BlockMatrix & matrices) const;

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
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src,
                                       unsigned int const                    problem_size) const;

protected:
  /*
   * These methods have to be overwritten by derived classes because these functions are
   * operator-specific and define how the operator looks like.
   */

  virtual void
  reinit_cell(unsigned int const cell) const;

  virtual void
  reinit_face(unsigned int const face) const;

  virtual void
  reinit_boundary_face(unsigned int const face) const;

  // standard integration procedure with separate loops for cell and face integrals
  virtual void
  do_cell_integral(IntegratorCell & integrator) const;

  virtual void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  virtual void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  virtual void
  do_boundary_integral_continuous(IntegratorFace &           integrator,
                                  OperatorType const &       operator_type,
                                  types::boundary_id const & boundary_id) const;

  // The computation of the diagonal and block-diagonal requires face integrals of type
  // interior (int) and exterior (ext)
  virtual void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  virtual void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  // cell-based computation of both cell and face integrals
  virtual void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

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
   * Data structure containing all operator-specific data.
   */
  AdditionalData data;

  /*
   * Matrix-free object.
   */
  mutable lazy_ptr<MatrixFree<dim, Number>> matrix_free;

  /*
   * Physical time (required for time-dependent problems).
   */
  mutable double time;

  /*
   * Constraint matrix.
   */
  mutable lazy_ptr<AffineConstraints<double>> constraint;

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

  std::shared_ptr<IntegratorCell> integrator;
  std::shared_ptr<IntegratorFace> integrator_m;
  std::shared_ptr<IntegratorFace> integrator_p;

  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef Elementwise::OperatorBase<dim, Number, This>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> ELEMENTWISE_PRECONDITIONER;
  typedef Elementwise::
    IterativeSolver<dim, n_components, Number, ELEMENTWISE_OPERATOR, ELEMENTWISE_PRECONDITIONER>
      ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR>       elementwise_operator;
  mutable std::shared_ptr<ELEMENTWISE_PRECONDITIONER> elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>         elementwise_solver;

private:
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
   * This function apply Dirichlet BCs for continuous Galerkin discretizations.
   */
  void
  cell_loop_dbc(MatrixFree<dim, Number> const & matrix_free,
                VectorType &                    dst,
                VectorType const &              src,
                Range const &                   range) const;

  /*
   * This function loops over all cells and calculates cell integrals.
   */
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   range) const;

  /*
   * This function loops over all interior faces and calculates face integrals.
   */
  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   range) const;

  /*
   * The following functions loop over all boundary faces and calculate boundary face integrals.
   * Depending on the operator type, we distinguish between boundary face integrals of type
   * homogeneous, inhomogeneous, and full.
   */

  // homogeneous operator
  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   range) const;

  // inhomogeneous operator
  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                    dst,
                                    VectorType const &              src,
                                    Range const &                   range) const;

  // full operator
  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   range) const;

  /*
   * inhomogeneous operator: For the inhomogeneous operator, we only have to calculate boundary face
   * integrals. The matrix-free implementation, however, does not offer interfaces for boundary face
   * integrals only. Hence we have to provide empty functions for cell and interior face integrals.
   */
  void
  cell_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const;

  void
  face_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const;

  /*
   * Calculate diagonal.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &              src,
                     Range const &                   range) const;

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &              src,
                     Range const &                   range) const;

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                              VectorType &                    dst,
                              VectorType const &              src,
                              Range const &                   range) const;

  void
  cell_based_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   range) const;

  /*
   * Calculate (assemble) block diagonal.
   */
  void
  cell_loop_block_diagonal(MatrixFree<dim, Number> const & matrix_free,
                           BlockMatrix &                   matrices,
                           BlockMatrix const &             src,
                           Range const &                   range) const;

  void
  face_loop_block_diagonal(MatrixFree<dim, Number> const & matrix_free,
                           BlockMatrix &                   matrices,
                           BlockMatrix const &             src,
                           Range const &                   range) const;

  void
  boundary_face_loop_block_diagonal(MatrixFree<dim, Number> const & matrix_free,
                                    BlockMatrix &                   matrices,
                                    BlockMatrix const &             src,
                                    Range const &                   range) const;

  // cell-based variant for computation of both cell and face integrals
  void
  cell_based_loop_block_diagonal(MatrixFree<dim, Number> const & matrix_free,
                                 BlockMatrix &                   matrices,
                                 BlockMatrix const &             src,
                                 Range const &                   range) const;

  /*
   * Apply block diagonal.
   */
  void
  cell_loop_apply_block_diagonal_matrix_based(MatrixFree<dim, Number> const & matrix_free,
                                              VectorType &                    dst,
                                              VectorType const &              src,
                                              Range const &                   range) const;

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal_matrix_based(MatrixFree<dim, Number> const & matrix_free,
                                                      VectorType &                    dst,
                                                      VectorType const &              src,
                                                      Range const &                   range) const;

#ifdef DEAL_II_WITH_TRILINOS
  /*
   * Calculate sparse matrix.
   */
  void
  cell_loop_calculate_system_matrix(MatrixFree<dim, Number> const & matrix_free,
                                    SparseMatrix &                  dst,
                                    SparseMatrix const &            src,
                                    Range const &                   range) const;

  void
  face_loop_calculate_system_matrix(MatrixFree<dim, Number> const & matrix_free,
                                    SparseMatrix &                  dst,
                                    SparseMatrix const &            src,
                                    Range const &                   range) const;

  void
  boundary_face_loop_calculate_system_matrix(MatrixFree<dim, Number> const & matrix_free,
                                             SparseMatrix &                  dst,
                                             SparseMatrix const &            src,
                                             Range const &                   range) const;
#endif

  /*
   * This function sets entries in the diagonal corresponding to constraint DoFs to one.
   */
  void
  set_constraint_diagonal(VectorType & diagonal) const;

  /*
   *  Verify that each boundary face is assigned exactly one boundary type.
   */
  void
  verify_boundary_conditions(
    DoFHandler<dim> const &                 dof_handler,
    std::vector<PeriodicFacePairIterator> & periodic_face_pairs_level0) const;

  /*
   *  Since the type of boundary conditions depends on the operator, this function has
   *  to be implemented by derived classes and can not be implemented in the abstract base class.
   */
  virtual void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                AdditionalData const &               operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  /*
   * Do we have to evaluate (boundary) face integrals for this operator? For example, operators
   * such as the mass matrix operator only involve cell integrals.
   */
  bool
  evaluate_face_integrals() const;

  /*
   * Multigrid level: 0 <= level <= max_level. If the operator is not used as a multigrid
   * level operator, this variable takes a value of numbers::invalid_unsigned_int.
   */
  unsigned int level;

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

  unsigned int n_mpi_processes;

  /*
   * for CG
   */
  std::vector<unsigned int>                      constrained_indices;
  mutable std::vector<std::pair<Number, Number>> constrained_values;
};

#include "operator_base.cpp"

#endif
