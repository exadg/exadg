#ifndef OPERATION_BASE_H
#define OPERATION_BASE_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "operator_type.h"

#include "../functionalities/lazy_ptr.h"
#include "multigrid_operator_base.h"

using namespace dealii;

template<int dim>
struct OperatorBaseData
{
  OperatorBaseData(const unsigned int dof_index,
                   const unsigned int quad_index,
                   const bool         cell_evaluate_values     = false,
                   const bool         cell_evaluate_gradients  = false,
                   const bool         cell_evaluate_hessians   = false,
                   const bool         cell_integrate_values    = false,
                   const bool         cell_integrate_gradients = false,
                   const bool         cell_integrate_hessians  = false,
                   const bool         face_evaluate_values     = false,
                   const bool         face_evaluate_gradients  = false,
                   const bool         face_integrate_values    = false,
                   const bool         face_integrate_gradients = false)
    : dof_index(dof_index),
      quad_index(quad_index),
      cell_evaluate(cell_evaluate_values, cell_evaluate_gradients, cell_evaluate_hessians),
      cell_integrate(cell_integrate_values, cell_integrate_gradients, cell_integrate_hessians),
      face_evaluate(face_evaluate_values, face_evaluate_gradients),
      face_integrate(face_integrate_values, face_integrate_gradients),
      use_cell_based_loops(false),
      evaluate_face_integrals(face_evaluate.do_eval() || face_integrate.do_eval()),
      operator_is_singular(false),
      mapping_update_flags(update_default),
      mapping_update_flags_inner_faces(update_default),
      mapping_update_flags_boundary_faces(update_default),
      implement_block_diagonal_preconditioner_matrix_free(false)
  {
  }

  struct Cell
  {
    Cell(const bool value = false, const bool gradient = false, const bool hessians = false)
      : value(value), gradient(gradient), hessians(hessians){};

    bool value;
    bool gradient;
    bool hessians;
  };

  struct Face
  {
    Face(const bool value = false, const bool gradient = false)
      : value(value), gradient(gradient){};

    bool
    do_eval() const
    {
      return value || gradient;
    }

    bool value;
    bool gradient;
  };

  template<typename Data>
  void
  append_mapping_update_flags(Data & other)
  {
    this->mapping_update_flags |= other.mapping_update_flags;
    this->mapping_update_flags_inner_faces |= other.mapping_update_flags_inner_faces;
    this->mapping_update_flags_boundary_faces |= other.mapping_update_flags_boundary_faces;
  }

  unsigned int dof_index;
  unsigned int quad_index;

  Cell cell_evaluate;
  Cell cell_integrate;
  Face face_evaluate;
  Face face_integrate;

  bool use_cell_based_loops;

  bool evaluate_face_integrals;

  bool operator_is_singular;

  UpdateFlags mapping_update_flags;
  UpdateFlags mapping_update_flags_inner_faces;
  UpdateFlags mapping_update_flags_boundary_faces;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs_level0;

  bool implement_block_diagonal_preconditioner_matrix_free;
};

template<int dim, int degree, typename Number, typename AdditionalData>
class OperatorBase : public MultigridOperatorBase<dim, Number>
{
public:
  static const int DIM = dim;

  typedef typename MultigridOperatorBase<dim, Number>::VectorType VectorType;

  typedef OperatorBase<dim, degree, Number, AdditionalData> This;

#ifdef DEAL_II_WITH_TRILINOS
  typedef FullMatrix<TrilinosScalar>     FullMatrix_;
  typedef TrilinosWrappers::SparseMatrix SparseMatrix;
#endif

  typedef std::vector<LAPACKFullMatrix<Number>>                BlockMatrix;
  typedef std::pair<unsigned int, unsigned int>                Range;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>     FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, 1, Number> FEEvalFace;
  typedef typename GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>
    PeriodicFacePairIterator;

  static const unsigned int vectorization_length = VectorizedArray<Number>::n_array_elements;
  static const unsigned int dofs_per_cell        = FEEvalCell::static_dofs_per_cell;

  OperatorBase();

  virtual ~OperatorBase()
  {
  }

  /*
   * If this method is called without the forth argument `level_mg_handler`, this operator is
   * initialized for level -1, i.e. the finest grid.
   */
  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         AdditionalData const &            operator_data,
         unsigned int                      level_mg_handler = numbers::invalid_unsigned_int) const;

  virtual void
  reinit(DoFHandler<dim> const &   dof_handler,
         Mapping<dim> const &      mapping,
         void *                    operator_data,
         MGConstrainedDoFs const & mg_constrained_dofs,
         unsigned int const        level_mg_handler);

  /*
   * Evaluate the homogeneous part of an operator. The homogeneous operator is the operator that is
   * obtained for homogeneous boundary conditions. This operation is typically applied in linear
   * iterative solvers (as well as multigrid preconditioners and smoothers). Operations of this type
   * are called apply_...() and vmult_...() as required by deal.II interfaces.
   */
  virtual void
  apply(VectorType & dst, VectorType const & src) const;

  virtual void
  apply_add(VectorType & dst, VectorType const & src, Number const time) const;

  virtual void
  apply_add(VectorType & dst, VectorType const & src) const;

  virtual void
  vmult(VectorType & dst, VectorType const & src) const;

  virtual void
  vmult_add(VectorType & dst, VectorType const & src) const;

  /*
   * Multigrid specific vmult_...() functions
   */
  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const;

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const;

  /*
   * evaluate inhomogeneous parts of operator related to inhomogeneous boundary face integrals.
   * Operations of this type are called rhs_...() since these functions are called to calculate the
   * vector forming the right-hand side vector of linear systems of equations.
   */
  void
  rhs(VectorType & dst) const;

  void
  rhs(VectorType & dst, Number const time) const;

  void
  rhs_add(VectorType & dst) const;

  void
  rhs_add(VectorType & dst, Number const time) const;

  /*
   * Evaluate the operator including homogeneous and inhomogeneous contributions. The typical use
   * case would be explicit time integration or the evaluation of nonlinear residuals where a
   * splitting into homogeneous and inhomogeneous contributions in not required.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const time) const;

  /*
   * point Jacobi preconditioner (diagonal)
   */
  virtual void
  calculate_diagonal(VectorType & diagonal) const;

  virtual void
  add_diagonal(VectorType & diagonal) const;

  virtual void
  add_diagonal(VectorType & diagonal, Number const time) const;

  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  /*
   * block Jacobi preconditioner (block-diagonal)
   */
  virtual void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

  // apply block diagonal elementwise: matrix-free implementation
  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src) const;

  // apply block diagonal elementwise: matrix-free implementation
  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src,
                                       Number const                          evaluation_time) const;

  // apply block diagonal: matrix-based implementation
  void
  apply_block_diagonal_matrix_based(VectorType & dst, VectorType const & src) const;

  // Update block diagonal preconditioner: initialize everything related to block diagonal
  // preconditioner when this function is called the first time. Recompute block matrices in case of
  // matrix-based implementation.
  virtual void
  update_block_diagonal_preconditioner() const;

  // This function has to initialize everything related to the block diagonal preconditioner when
  // using the matrix-free variant with elementwise iterative solvers and matrix-free operator
  // evaluation.
  virtual void
  initialize_block_diagonal_preconditioner_matrix_free() const
  {
    AssertThrow(
      false,
      ExcMessage(
        "Should not arrive here. Function initialize_block_diagonal_preconditioner_matrix_free() has to be implemented by derived classes."));
  }

  virtual void
  add_block_diagonal_matrices(BlockMatrix & matrices) const;

  virtual void
  add_block_diagonal_matrices(BlockMatrix & matrices, Number const time) const;

  /*
   * sparse matrix (Trilinos) methods
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(SparseMatrix & system_matrix) const;

  virtual void
  calculate_system_matrix(SparseMatrix & system_matrix) const;

  virtual void
  calculate_system_matrix(SparseMatrix & system_matrix, Number const time) const;
#endif

  /*
   * Utility functions required by deal.II as interfaces.
   */
  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  Number
  el(unsigned int const, unsigned int const) const;

  bool
  is_empty_locally() const;

  /*
   *  Getters and setters.
   */
  const MatrixFree<dim, Number> &
  get_data() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  AdditionalData const &
  get_operator_data() const;

  void
  set_evaluation_time(double const evaluation_time_in) const;

  double
  get_evaluation_time() const;

  unsigned int
  get_level() const;

  AffineConstraints<double> const &
  get_constraint_matrix() const;

  /*
   * Initializes a dof-vector.
   */
  void
  initialize_dof_vector(VectorType & vector) const;

  /*
   * Returns whether the operator is singular, e.g., the Laplace operator with pure Neumann boundary
   * conditions is singular.
   */
  bool
  is_singular() const;

protected:
  /*
   * These methods have to be overwritten by derived classes because these functions are
   * operator-specific and define how the operator looks like.
   */
  virtual void
  do_cell_integral(FEEvalCell & /*fe_eval*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_cell_integral() has not been implemented!"));
  }

  virtual void
  do_face_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_face_integral() has not been implemented!"));
  }

  virtual void
  do_face_int_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false,
                ExcMessage("OperatorBase::do_face_int_integral() has not been implemented!"));
  }

  virtual void
  do_face_ext_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false,
                ExcMessage("OperatorBase::do_face_ext_integral() has not been implemented!"));
  }

  virtual void
  do_boundary_integral(FEEvalFace & /*fe_eval*/,
                       OperatorType const & /*operator_type*/,
                       types::boundary_id const & /*boundary_id*/) const
  {
    AssertThrow(false,
                ExcMessage("OperatorBase::do_boundary_integral() has not been implemented!"));
  }

  virtual void
  do_block_diagonal_cell_based() const
  {
    AssertThrow(
      false, ExcMessage("OperatorBase::do_block_diagonal_cell_based() has not been implemented!"));
  }

  /*
   * Data structure containing all operator-specific data.
   */
  mutable AdditionalData operator_data;

  /*
   * Matrix-free object.
   */
  mutable lazy_ptr<MatrixFree<dim, Number>> data;

  /*
   * Evaluation time (required for time-dependent problems).
   */
  mutable double eval_time;

private:
  /*
   * Helper functions:
   *
   * The diagonal, block-diagonal, as well as the system matrix (assembled into a sparse matrix) are
   * computed columnwise. This means that column i of the block-matrix is computed by evaluating the
   * operator for a unit vector which takes a value of 1 in row i and is 0 for all other entries.
   */
  void
  create_standard_basis(unsigned int j, FEEvalCell & fe_eval) const;

  void
  create_standard_basis(unsigned int j, FEEvalFace & fe_eval) const;

  void
  create_standard_basis(unsigned int j, FEEvalFace & fe_eval_1, FEEvalFace & fe_eval_2) const;

  /*
   * This function loops over all cells and calculates cell integrals.
   */
  void
  cell_loop(MatrixFree<dim, Number> const & /*data*/,
            VectorType &       dst,
            VectorType const & src,
            Range const &      range) const;

  /*
   * This function loops over all interior faces and calculates face integrals.
   */
  void
  face_loop(MatrixFree<dim, Number> const & /*data*/,
            VectorType &       dst,
            VectorType const & src,
            Range const &      range) const;

  /*
   * The following functions loop over all boundary faces and calculate boundary face integrals.
   * Depending on the operator type, we distinguish between boundary face integrals of type
   * homogeneous, inhomogeneous, and full.
   */

  // homogeneous operator
  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & /*data*/,
                                  VectorType & /*dst*/,
                                  VectorType const & /*src*/,
                                  Range const & /*range*/) const;

  // inhomogeneous operator
  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & /*data*/,
                                    VectorType & /*dst*/,
                                    VectorType const & /*src*/,
                                    Range const & /*range*/) const;

  // full operator
  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & /*data*/,
                                   VectorType & /*dst*/,
                                   VectorType const & /*src*/,
                                   Range const & /*range*/) const;

  /*
   * inhomogeneous operator: For the inhomogeneous operator, we only have to calculate boundary face
   * integrals. The matrix-free implementation, however, does not offer interfaces for boundary face
   * integrals only. Hence we have to provide empty functions for cell and interior face integrals.
   */
  void
  cell_loop_empty(MatrixFree<dim, Number> const & /*data*/,
                  VectorType & /*dst*/,
                  VectorType const & /*src*/,
                  Range const & /*range*/) const
  {
    // nothing to do
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const & /*data*/,
                  VectorType & /*dst*/,
                  VectorType const & /*src*/,
                  Range const & /*range*/) const
  {
    // nothing to do
  }

  /*
   * Calculate diagonal.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & /*data*/,
                     VectorType & dst,
                     VectorType const & /*src*/,
                     Range const & range) const;

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & /*data*/,
                     VectorType & dst,
                     VectorType const & /*src*/,
                     Range const & range) const;

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & /*data*/,
                              VectorType & dst,
                              VectorType const & /*src*/,
                              Range const & range) const;

  void
  cell_based_loop_diagonal(MatrixFree<dim, Number> const & /*data*/,
                           VectorType & dst,
                           VectorType const & /*src*/,
                           Range const & range) const;

  /*
   * Calculate (assemble) block diagonal.
   */
  void
  cell_loop_block_diagonal(MatrixFree<dim, Number> const & data,
                           BlockMatrix &                   matrices,
                           BlockMatrix const & /*src*/,
                           Range const & range) const;

  void
  face_loop_block_diagonal(MatrixFree<dim, Number> const & data,
                           BlockMatrix &                   matrices,
                           BlockMatrix const & /*src*/,
                           Range const & range) const;

  void
  boundary_face_loop_block_diagonal(MatrixFree<dim, Number> const & data,
                                    BlockMatrix &                   matrices,
                                    BlockMatrix const & /*src*/,
                                    Range const & range) const;

  // cell-based variant for computation of both cell and face integrals
  void
  cell_based_loop_block_diagonal(MatrixFree<dim, Number> const & data,
                                 BlockMatrix &                   matrices,
                                 BlockMatrix const & /*src*/,
                                 Range const & range) const;


  /*
   * Apply block diagonal.
   */
  void
  cell_loop_apply_block_diagonal_matrix_based(MatrixFree<dim, Number> const & /*data*/,
                                              VectorType &       dst,
                                              VectorType const & src,
                                              Range const &      range) const;

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   cell_range) const;

#ifdef DEAL_II_WITH_TRILINOS
  /*
   * Calculate sparse matrix.
   */
  void
  cell_loop_calculate_system_matrix(MatrixFree<dim, Number> const & /*data*/,
                                    SparseMatrix & dst,
                                    SparseMatrix const & /*src*/,
                                    Range const & range) const;

  void
  face_loop_calculate_system_matrix(MatrixFree<dim, Number> const & /*data*/,
                                    SparseMatrix & dst,
                                    SparseMatrix const & /*src*/,
                                    Range const & range) const;

  void
  boundary_face_loop_calculate_system_matrix(MatrixFree<dim, Number> const & /*data*/,
                                             SparseMatrix & /*dst*/,
                                             SparseMatrix const & /*src*/,
                                             Range const & /*range*/) const;
#endif

  /*
   * For singular operators, a Krylov projection is applied onto the subspace of vectors with zero
   * mean. This function calculates the diagonal for the projected system given the diagonal of the
   * original system.
   */
  void
  adjust_diagonal_for_singular_operator(VectorType & diagonal) const;

  /*
   * This function sets entries in the diagonal corresponding to constraint DoFs to one.
   */
  void
  set_constraint_diagonal(VectorType & diagonal) const;

  void
  add_constraints(DoFHandler<dim> const &     dof_handler,
                  AffineConstraints<double> & constraint_own,
                  MGConstrainedDoFs const &   mg_constrained_dofs,
                  AdditionalData &            operator_data,
                  unsigned int const          level);

  /*
   * Add periodic constraints: loop over all periodic face pairs on level 0
   */
  void
  add_periodicity_constraints(DoFHandler<dim> const &                 dof_handler,
                              unsigned int const                      level,
                              std::vector<PeriodicFacePairIterator> & periodic_face_pairs_level0,
                              AffineConstraints<double> &             constraint_own);

  /*
   * Add periodic constraints: for a given face pair on level 0 add recursively
   * all subfaces on the given level
   */
  void
  add_periodicity_constraints(unsigned int const                            level,
                              unsigned int const                            target_level,
                              typename DoFHandler<dim>::face_iterator const face1,
                              typename DoFHandler<dim>::face_iterator const face2,
                              AffineConstraints<double> &                   constraints);

  /*
   *  Verify that each boundary face is assigned exactly one boundary type.
   */
  void
  verify_boundary_conditions(DoFHandler<dim> const & dof_handler,
                             AdditionalData const &  operator_data) const;

  /*
   *  Since the type of boundary conditions depends on the operator, this function has
   *  to be implemented by derived classes and can not be implemented in the abstract base class.
   */
  virtual void
  do_verify_boundary_conditions(
    types::boundary_id const /* boundary_id */,
    AdditionalData const & /* operator_data */,
    std::set<types::boundary_id> const & /* periodic_boundary_ids */) const
  {
    AssertThrow(
      false,
      ExcMessage(
        "OperatorBase::do_verify_boundary_conditions() has to be implemented by derived classes."));
  }

  /*
   * Do we have to evaluate (boundary) face integrals for this operator? For example, ome operators
   * such as the mass matrix operator only involve cell integrals.
   */
  const bool do_eval_faces;

  /*
   * Constraint matrix.
   */
  mutable lazy_ptr<AffineConstraints<double>> constraint;

  /*
   * Is the discretization based on discontinuous Galerin method?
   */
  mutable bool is_dg;

  /*
   * Operator is used as a multigrid level operator?
   */
  mutable bool is_mg;

  /*
   * Multigrid level: 0 <= level_mg_handler <= max_level. If the operator is not used as a multigrid
   * level operator, this variable takes a value of numbers::invalid_unsigned_int.
   */
  mutable unsigned int level_mg_handler;

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

  // FEEvaluation objects required for elementwise application of block Jacobi operation
  mutable std::shared_ptr<FEEvalCell> fe_eval;
  mutable std::shared_ptr<FEEvalFace> fe_eval_m;
  mutable std::shared_ptr<FEEvalFace> fe_eval_p;
};

#include "operator_base.cpp"

#endif
