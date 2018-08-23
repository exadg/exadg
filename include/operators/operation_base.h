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

#include "multigrid_operator_base.h"

using namespace dealii;

template<int dim, typename BT, typename OT, typename BoundaryDescriptor>
struct OperatorBaseData
{
  typedef BT BoundaryType;
  typedef OT OperatorType;

  OperatorBaseData(const unsigned int dof_index,
                   const unsigned int quad_index,
                   const bool         cell_evaluate_values = false,
                   const bool         cell_evaluate_gradients = false,
                   const bool         cell_evaluate_hessians = false,
                   const bool         cell_integrate_values = false,
                   const bool         cell_integrate_gradients = false,
                   const bool         cell_integrate_hessians = false,
                   const bool         face_evaluate_values = false,
                   const bool         face_evaluate_gradients = false,
                   const bool         face_integrate_values = false,
                   const bool         face_integrate_gradients = false,
                   const bool         boundary_evaluate_values = false,
                   const bool         boundary_evaluate_gradients = false,
                   const bool         boundary_integrate_values = false,
                   const bool         boundary_integrate_gradients = false)
    : dof_index(dof_index),
      quad_index(quad_index),
      cell_evaluate(cell_evaluate_values, cell_evaluate_gradients, cell_evaluate_hessians),
      cell_integrate(cell_integrate_values, cell_integrate_gradients, cell_integrate_hessians),
      internal_evaluate(face_evaluate_values, face_evaluate_gradients),
      internal_integrate(face_integrate_values, face_integrate_gradients),
      boundary_evaluate(boundary_evaluate_values, boundary_evaluate_gradients),
      boundary_integrate(boundary_integrate_values, boundary_integrate_gradients),
      use_cell_based_loops(false),
      operator_is_singular(false)
  {
  }

  struct Cell
  {
    Cell(const bool value = false, const bool gradient = false, const bool hessians = false)
      : value(value), gradient(gradient), hessians(hessians){};
    /*const*/ bool value;
    /*const*/ bool gradient;
    /*const*/ bool hessians;
  };

  struct Face
  {
    Face(const bool value = false, const bool gradient = false) : value(value), gradient(gradient){};
    /*const*/ bool value;
    /*const*/ bool gradient;

    bool
    do_eval() const
    {
      return value || gradient;
    }
  };

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(bc->dirichlet_bc.find(boundary_id) != bc->dirichlet_bc.end())
      return BoundaryType::dirichlet;
    else if(bc->neumann_bc.find(boundary_id) != bc->neumann_bc.end())
      return BoundaryType::neumann;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::undefined;
  }

  /*const*/ unsigned int dof_index;
  /*const*/ unsigned int quad_index;

  /*const*/ Cell cell_evaluate;
  /*const*/ Cell cell_integrate;
  /*const*/ Face internal_evaluate;
  /*const*/ Face internal_integrate;
  /*const*/ Face boundary_evaluate;
  /*const*/ Face boundary_integrate;

  bool use_cell_based_loops;

  bool operator_is_singular;

  std::shared_ptr<BoundaryDescriptor> bc;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs_level0;
};

template<typename T>
class lazy_ptr
{
public:
  lazy_ptr() : tp(&t)
  {
  }

  void
  reset()
  {
    this->tp = &this->t;
  }
  
  void
  reinit(T const & t)
  {
    this->tp = &t;
  }
  
  T &
  own()
  {
    return t;
  }
  
  T const * operator->()
  {
    return tp;
  }
  
  T const & operator*()
  {
    return *tp;
  }

private:
  T         t;
  T const * tp;
};

template<int dim, int degree, typename Number, typename AdditionalData>
class OperatorBase : public MultigridOperatorBase<dim, Number>
{
public:
  static const int                                          DIM = dim;
  typedef OperatorBase<dim, degree, Number, AdditionalData> This;
  typedef parallel::distributed::Vector<Number>             VectorType;
#ifdef DEAL_II_WITH_TRILINOS
  typedef FullMatrix<TrilinosScalar>     FullMatrix_;
  typedef TrilinosWrappers::SparseMatrix SparseMatrix;
#endif
  typedef std::vector<LAPACKFullMatrix<Number>>                BlockMatrix;
  typedef MatrixFree<dim, Number>                              MatrixFree_;
  typedef std::pair<unsigned int, unsigned int>                Range;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>     FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, 1, Number> FEEvalFace;
  typedef typename AdditionalData::BoundaryType                BoundaryType;
  typedef typename AdditionalData::OperatorType                OperatorType;

  OperatorBase();

  static const unsigned int vectorization_length = VectorizedArray<Number>::n_array_elements;
  static const unsigned int dofs_per_cell        = FEEvalCell::static_dofs_per_cell;

  void
  reinit(MatrixFree_ const &    matrix_free,
         ConstraintMatrix &     constraint_matrix,
         AdditionalData const & operator_settings,
         unsigned int           level_mg_handler = numbers::invalid_unsigned_int) const;

  virtual void
  reinit(const DoFHandler<dim> &   dof_handler,
         const Mapping<dim> &      mapping,
         void *                    operator_settings,
         const MGConstrainedDoFs & mg_constrained_dofs,
         const unsigned int        level);

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
  apply_block_jacobi(VectorType & dst, VectorType const & src) const;
  
  void
  apply_block_jacobi_add(VectorType & dst, VectorType const & src) const;

  // TODO: add matrix-free and block matrix version
  void
  apply_block_diagonal(VectorType & dst, VectorType const & src) const;
  
  void
  update_block_jacobi() const;
  
  void
  update_block_jacobi(bool const do_lu_factorization) const;
  
  virtual void
  add_block_jacobi_matrices(BlockMatrix & matrices) const;
  
  virtual void
  add_block_jacobi_matrices(BlockMatrix & matrices, Number const time) const;

  /*
   * sparse matrix (Trilinos) methods
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(SparseMatrix & system_matrix) const;
  void
  calculate_system_matrix(SparseMatrix & system_matrix) const;
#endif

  /*
   * utility functions
   */
  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  Number
  el(const unsigned int, const unsigned int) const;

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

protected:
  /*
   * methods to be overwritten
   */
  virtual void
  do_cell_integral(FEEvalCell & /*fe_eval*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_cell_integral has not been implemented!"));
  }
  
  virtual void
  do_face_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_face_integral has not been implemented!"));
  }
  
  virtual void
  do_face_int_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_face_int_integral has not been implemented!"));
  }
  
  virtual void
  do_face_ext_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & /*fe_eval_p*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_face_ext_integral has not been implemented!"));
  }
  
  virtual void
  do_boundary_integral(FEEvalFace & /*fe_eval*/,
                       OperatorType const & /*operator_type*/,
                       types::boundary_id const & /*boundary_id*/) const
  {
    AssertThrow(false, ExcMessage("OperatorBase::do_boundary_integral has not been implemented!"));
  }

  /*
   * helper functions
   */
  void
  create_standard_basis(unsigned int j, FEEvalCell & fe_eval) const;

  void
  create_standard_basis(unsigned int j, FEEvalFace & fe_eval) const;

  void
  create_standard_basis(unsigned int j, FEEvalFace & fe_eval1, FEEvalFace & fe_eval2) const;

  /*
   * functions to be called from matrix-free loops and cell_loops: vmult (homogenous)
   */
  void
  local_cell_hom(const MatrixFree_ & /*data*/, VectorType & dst, const VectorType & src, const Range & range) const;

  void
  local_face_hom(const MatrixFree_ & /*data*/, VectorType & dst, const VectorType & src, const Range & range) const;

  void
  local_boundary_hom(const MatrixFree_ & /*data*/,
                     VectorType & /*dst*/,
                     const VectorType & /*src*/,
                     const Range & /*range*/) const;

  /*
   * ... rhs (inhomogenous)
   * note:  in the inhomogeneous case we only have to loop over the boundary 
   * faces. This is, however, not possible with Matrixfree::loop(): that is 
   * why two empty function have to be provided for cell and face.
   */
  void
  local_cell_inhom(const MatrixFree_ & /*data*/, VectorType & /*dst*/, const VectorType & /*src*/, const Range & /*range*/) const
  {
    // nothing to do
  }

  void
  local_face_inhom(const MatrixFree_ & /*data*/, VectorType & /*dst*/, const VectorType & /*src*/, const Range & /*range*/) const
  {
    // nothing to do
  }
  
  void
  local_boundary_inhom(const MatrixFree_ & /*data*/,
                       VectorType & /*dst*/,
                       const VectorType & /*src*/,
                       const Range & /*range*/) const;
  
  /*
   * ... evaluate
   */
  void
  local_boundary_full(const MatrixFree_ & /*data*/,
                      VectorType & /*dst*/,
                      const VectorType & /*src*/,
                      const Range & /*range*/) const;

  /*
   * ... diagonal
   */
  void
  local_add_diagonal_cell(const MatrixFree_ & /*data*/,
                          VectorType & dst,
                          const VectorType & /*src*/,
                          const Range & range) const;

  void
  local_add_diagonal_face(const MatrixFree_ & /*data*/,
                          VectorType & dst,
                          const VectorType & /*src*/,
                          const Range & range) const;

  void
  local_add_diagonal_boundary(const MatrixFree_ & /*data*/,
                              VectorType & dst,
                              const VectorType & /*src*/,
                              const Range & range) const;

  void
  local_add_diagonal_cell_based(const MatrixFree_ & /*data*/,
                                VectorType & dst,
                                const VectorType & /*src*/,
                                const Range & range) const;

  /*
   * ... block diagonal
   */
  void
  local_apply_block_diagonal(const MatrixFree_ & /*data*/,
                             VectorType &       dst,
                             const VectorType & src,
                             const Range &      range) const;
  void
  local_add_block_diagonal_cell(const MatrixFree_ & /*data*/,
                                BlockMatrix & dst,
                                const BlockMatrix & /*src*/,
                                const Range & range) const;

  void
  local_add_block_diagonal_face(const MatrixFree_ & /*data*/,
                                BlockMatrix & dst,
                                const BlockMatrix & /*src*/,
                                const Range & range) const;

  void
  local_add_block_diagonal_boundary(const MatrixFree_ & /*data*/,
                                    BlockMatrix & dst,
                                    const BlockMatrix & /*src*/,
                                    const Range & range) const;

  void
  local_add_block_diagonal_cell_based(const MatrixFree_ & /*data*/,
                                      BlockMatrix & dst,
                                      const BlockMatrix & /*src*/,
                                      const Range & range) const;

  /*
   * ... block Jacobi (inverse of block diagonal)
   * same as local_apply_block_diagonal, but instead of applying the block matrix B
   * we solve the linear system B*dst=src (LU factorization should have already
   * been performed with the method update_block_jacobi(true))
   */
  void
  local_apply_block_jacobi_add(const MatrixFree_ &         data,
                               VectorType &       dst,
                               const VectorType & src,
                               const Range &      cell_range) const;

  /*
   * ... sparse matrix
   */
#ifdef DEAL_II_WITH_TRILINOS
  void
  local_calculate_system_matrix_cell(const MatrixFree_ & /*data*/,
                                     SparseMatrix & dst,
                                     const SparseMatrix & /*src*/,
                                     const Range & range) const;

  void
  local_calculate_system_matrix_face(const MatrixFree_ & /*data*/,
                                     SparseMatrix & dst,
                                     const SparseMatrix & /*src*/,
                                     const Range & range) const;

  void
  local_calculate_system_matrix_boundary(const MatrixFree_ & /*data*/,
                                         SparseMatrix & /*dst*/,
                                         const SparseMatrix & /*src*/,
                                         const Range & /*range*/) const;
#endif

protected:
  void
  adjust_diagonal_for_singular_operator(VectorType & diagonal) const;

  void
  set_constraint_diagonal(VectorType & diagonal) const;

  void
  add_periodicity_constraints(const unsigned int                            level,
                              const unsigned int                            target_level,
                              const typename DoFHandler<dim>::face_iterator face1,
                              const typename DoFHandler<dim>::face_iterator face2,
                              ConstraintMatrix &                            constraints);

  bool
  verify_boundary_conditions(DoFHandler<dim> const & dof_handler, AdditionalData const & operator_data);

protected:
  mutable AdditionalData operator_settings;
  mutable lazy_ptr<MatrixFree_> data;
  mutable double eval_time;

private:
  const bool                         do_eval_faces;
  mutable lazy_ptr<ConstraintMatrix> constraint;
  mutable bool                       is_dg;
  mutable bool                       is_mg;
  mutable unsigned int               level_mg_handler;

  mutable std::vector<LAPACKFullMatrix<Number>> matrices;

  mutable bool block_jacobi_matrices_have_been_initialized;

};

#include "operation_base.cpp"

#endif
