#ifndef OPERATION_BASE_H
#define OPERATION_BASE_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/parallel_vector.h>
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "operator_type.h"

#include "../functionalities/lazy_ptr.h"
#include "multigrid_operator_base.h"

using namespace dealii;

template<int dim, typename BoundaryDescriptor>
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
      mapping_update_flags_boundary_faces(update_default)
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

  std::shared_ptr<BoundaryDescriptor> bc;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs_level0;
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
  typedef MatrixFree<dim, Number>                              MatrixFree_;
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

  // if this method is called without the forth argument `level_mg_handler`,
  // this operator is initialized for level -1, i.e. the finest grid
  void
  reinit(MatrixFree_ const &      matrix_free,
         ConstraintMatrix const & constraint_matrix,
         AdditionalData const &   operator_data,
         unsigned int             level_mg_handler = numbers::invalid_unsigned_int) const;

  virtual void
  reinit(DoFHandler<dim> const &   dof_handler,
         Mapping<dim> const &      mapping,
         void *                    operator_data,
         MGConstrainedDoFs const & mg_constrained_dofs,
         unsigned int const        level_mg_handler);

  /*
   * matrix vector multiplication
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

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const;

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const;

  /*
   *
   */
  void
  rhs(VectorType & dst) const;

  void
  rhs(VectorType & dst, Number const time) const;

  void
  rhs_add(VectorType & dst) const;

  void
  rhs_add(VectorType & dst, Number const time) const;

  void
  evaluate(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const time) const;

  /*
   * point Jacobi method
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
   * block Jacobi methods
   */
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

  // TODO: add matrix-free and block matrix version
  void
  apply_block_diagonal(VectorType & dst, VectorType const & src) const;

  void
  update_inverse_block_diagonal() const;

  void
  calculate_block_diagonal_matrices() const;

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
   * utility functions
   */
  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  Number
  el(unsigned int const, unsigned int const) const;

  bool
  is_empty_locally() const;

  const MatrixFree<dim, Number> &
  get_data() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  AdditionalData const &
  get_operator_data() const;

  void
  initialize_dof_vector(VectorType & vector) const;

  void
  set_evaluation_time(double const evaluation_time_in) const;

  double
  get_evaluation_time() const;

  bool
  is_singular() const;

  unsigned int
  get_level() const
  {
    return level_mg_handler;
  }

  ConstraintMatrix const &
  get_constraint_matrix() const
  {
    return *constraint;
  }

protected:
  /*
   * methods to be overwritten
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

  mutable AdditionalData operator_data;

  mutable lazy_ptr<MatrixFree_> data;

  mutable double eval_time;

private:
  /*
   * helper functions
   */
  template<typename FEEval>
  void
  create_standard_basis(unsigned int j, FEEval & fe_eval) const
  {
    // create a standard basis in the dof values of FEEvalution
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
    fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
  }

  void
  create_standard_basis(unsigned int j, FEEvalFace & fe_eval1, FEEvalFace & fe_eval2) const;

  /*
   * functions to be called from matrix-free loops and cell_loops: vmult (homogenous)
   */
  void
  local_cell_hom(MatrixFree_ const & /*data*/,
                 VectorType &       dst,
                 VectorType const & src,
                 Range const &      range) const;

  void
  local_face_hom(MatrixFree_ const & /*data*/,
                 VectorType &       dst,
                 VectorType const & src,
                 Range const &      range) const;

  void
  local_boundary_hom(MatrixFree_ const & /*data*/,
                     VectorType & /*dst*/,
                     VectorType const & /*src*/,
                     Range const & /*range*/) const;

  /*
   * ... rhs (inhomogenous)
   * note:  in the inhomogeneous case we only have to loop over the boundary
   * faces. This is, however, not possible with Matrixfree::loop(): that is
   * why two empty function have to be provided for cell and face.
   */
  void
  local_cell_inhom(MatrixFree_ const & /*data*/,
                   VectorType & /*dst*/,
                   VectorType const & /*src*/,
                   Range const & /*range*/) const
  {
    // nothing to do
  }

  void
  local_face_inhom(MatrixFree_ const & /*data*/,
                   VectorType & /*dst*/,
                   VectorType const & /*src*/,
                   Range const & /*range*/) const
  {
    // nothing to do
  }

  void
  local_boundary_inhom(MatrixFree_ const & /*data*/,
                       VectorType & /*dst*/,
                       VectorType const & /*src*/,
                       Range const & /*range*/) const;

  /*
   * ... evaluate
   */
  void
  local_boundary_full(MatrixFree_ const & /*data*/,
                      VectorType & /*dst*/,
                      VectorType const & /*src*/,
                      Range const & /*range*/) const;

  /*
   * ... diagonal
   */
  void
  local_add_diagonal_cell(MatrixFree_ const & /*data*/,
                          VectorType & dst,
                          VectorType const & /*src*/,
                          Range const & range) const;

  void
  local_add_diagonal_face(MatrixFree_ const & /*data*/,
                          VectorType & dst,
                          VectorType const & /*src*/,
                          Range const & range) const;

  void
  local_add_diagonal_boundary(MatrixFree_ const & /*data*/,
                              VectorType & dst,
                              VectorType const & /*src*/,
                              Range const & range) const;

  void
  local_add_diagonal_cell_based(MatrixFree_ const & /*data*/,
                                VectorType & dst,
                                VectorType const & /*src*/,
                                Range const & range) const;

  /*
   * ... block diagonal
   */
  void
  local_apply_block_diagonal(MatrixFree_ const & /*data*/,
                             VectorType &       dst,
                             VectorType const & src,
                             Range const &      range) const;
  void
  local_add_block_diagonal_cell(MatrixFree_ const & /*data*/,
                                BlockMatrix & dst,
                                BlockMatrix const & /*src*/,
                                Range const & range) const;

  void
  local_add_block_diagonal_face(MatrixFree_ const & /*data*/,
                                BlockMatrix & dst,
                                BlockMatrix const & /*src*/,
                                Range const & range) const;

  void
  local_add_block_diagonal_boundary(MatrixFree_ const & /*data*/,
                                    BlockMatrix & dst,
                                    BlockMatrix const & /*src*/,
                                    Range const & range) const;

  void
  local_add_block_diagonal_cell_based(MatrixFree_ const & /*data*/,
                                      BlockMatrix & dst,
                                      BlockMatrix const & /*src*/,
                                      Range const & range) const;

  /*
   * ... block Jacobi (inverse of block diagonal)
   * same as local_apply_block_diagonal, but instead of applying the block matrix B
   * we solve the linear system B*dst=src (LU factorization should have already
   * been performed with the method update_inverse_block_diagonal())
   */
  void
  local_apply_inverse_block_diagonal(MatrixFree_ const & data,
                                     VectorType &        dst,
                                     VectorType const &  src,
                                     Range const &       cell_range) const;

  /*
   * ... sparse matrix
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  local_calculate_system_matrix_cell(MatrixFree_ const & /*data*/,
                                     SparseMatrix & dst,
                                     SparseMatrix const & /*src*/,
                                     Range const & range) const;

  void
  local_calculate_system_matrix_face(MatrixFree_ const & /*data*/,
                                     SparseMatrix & dst,
                                     SparseMatrix const & /*src*/,
                                     Range const & range) const;

  void
  local_calculate_system_matrix_boundary(MatrixFree_ const & /*data*/,
                                         SparseMatrix & /*dst*/,
                                         SparseMatrix const & /*src*/,
                                         Range const & /*range*/) const;
#endif

  void
  adjust_diagonal_for_singular_operator(VectorType & diagonal) const;

  /*
   * set entries in the diagonal corresponding to constraint DoFs to one
   */
  void
  set_constraint_diagonal(VectorType & diagonal) const;

  void
  add_constraints(DoFHandler<dim> const &   dof_handler,
                  ConstraintMatrix &        constraint_own,
                  MGConstrainedDoFs const & mg_constrained_dofs,
                  AdditionalData &          operator_data,
                  unsigned int const        level);

  /*
   * Add periodic constraints: loop over all periodic face pairs on level 0
   */
  void
  add_periodicity_constraints(DoFHandler<dim> const &                 dof_handler,
                              unsigned int const                      level,
                              std::vector<PeriodicFacePairIterator> & periodic_face_pairs_level0,
                              ConstraintMatrix &                      constraint_own);

  /*
   * Add periodic constraints: for a given face pair on level 0 add recursively
   * all subfaces on the given level
   */
  void
  add_periodicity_constraints(unsigned int const                            level,
                              unsigned int const                            target_level,
                              typename DoFHandler<dim>::face_iterator const face1,
                              typename DoFHandler<dim>::face_iterator const face2,
                              ConstraintMatrix &                            constraints);

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
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                AdditionalData const &               operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const
  {
    AssertThrow(
      false,
      ExcMessage(
        "OperatorBase::do_verify_boundary_conditions() has to be implemented by derived classes."));
  }

  // do we have to evaluate (boundary) face integrals for this operator? For example,
  // some operators such as the mass matrix operator only involve cell integrals.
  const bool do_eval_faces;

  mutable lazy_ptr<ConstraintMatrix> constraint;

  // discretization is based on discontinuous Galerin method?
  mutable bool is_dg;

  // operator is used as a multigrid level operator?
  mutable bool is_mg;

  mutable unsigned int level_mg_handler;

  // vector of matrices for block-diagonal preconditioners
  mutable std::vector<LAPACKFullMatrix<Number>> matrices;
  mutable bool                                  block_jacobi_matrices_have_been_initialized;
};

#include "operation_base.cpp"

#endif
