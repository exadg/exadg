#ifndef OPERATION_BASE_H
#define OPERATION_BASE_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

template <int dim, typename BT, typename OT, typename BoundaryDescriptor>
struct OperatorBaseData {

  typedef BT BoundaryType;
  typedef OT OperatorType;

  OperatorBaseData(const unsigned int dof_index, const unsigned int quad_index,
                   const bool c_e_v = false, const bool c_e_g = false,
                   const bool c_e_h = false, const bool c_i_v = false,
                   const bool c_i_g = false, const bool c_i_h = false,
                   const bool f_e_v = false, const bool f_e_g = false,
                   const bool f_i_v = false, const bool f_i_g = false,
                   const bool b_e_v = false, const bool b_e_g = false,
                   const bool b_i_v = false, const bool b_i_g = false)
      : dof_index(dof_index), quad_index(quad_index),
        cell_evaluate(c_e_v, c_e_g, c_e_h), cell_integrate(c_i_v, c_i_g, c_i_h),
        internal_evaluate(f_e_v, f_e_g), internal_integrate(f_i_v, f_i_g),
        boundary_evaluate(b_e_v, b_e_g), boundary_integrate(b_i_v, b_i_g) {}

  struct Cell {
    Cell(const bool value = false, const bool gradient = false,
         const bool hessians = false)
        : value(value), gradient(gradient), hessians(hessians){};
    /*const*/ bool value;
    /*const*/ bool gradient;
    /*const*/ bool hessians;
  };

  struct Face {
    Face(const bool value = false, const bool gradient = false)
        : value(value), gradient(gradient){};
    /*const*/ bool value;
    /*const*/ bool gradient;

    bool do_eval() const { return value || gradient; }
  };

  inline DEAL_II_ALWAYS_INLINE BoundaryType
  get_boundary_type(types::boundary_id const &boundary_id) const {

    if (bc->dirichlet_bc.find(boundary_id) != bc->dirichlet_bc.end())
      return BoundaryType::dirichlet;
    else if (bc->neumann_bc.find(boundary_id) != bc->neumann_bc.end())
      return BoundaryType::neumann;

    AssertThrow(
        false,
        ExcMessage("Boundary type of face is invalid or not implemented."));

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
  std::shared_ptr<BoundaryDescriptor> bc;
};

template <typename T> class LazyWrapper {

public:
  void reinit(T &t) { this->tp = &t; }
  T &own() { return t; }
  T *operator->() { return tp; }
  T &operator*() { return *tp; }

private:
  T *tp;
  T t;
};

template <int dim, int degree, typename Number, typename AdditionalData>
class OperatorBase : public MatrixOperatorBaseNew<dim, Number> {

public:
  typedef OperatorBase<dim, degree, Number, AdditionalData> This;
  typedef parallel::distributed::Vector<Number> VNumber;
  typedef FullMatrix<Number> FMatrix;
  typedef std::vector<LAPACKFullMatrix<Number>> BMatrix;
  typedef TrilinosWrappers::SparseMatrix SMatrix;
  typedef MatrixFree<dim, Number> MF;
  typedef ConstraintMatrix CM;
  typedef std::pair<unsigned int, unsigned int> Range;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number> FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, 1, Number> FEEvalFace;
  typedef typename AdditionalData::BoundaryType BoundaryType;
  typedef typename AdditionalData::OperatorType OperatorType;

  OperatorBase();

  static const unsigned int v_len = VectorizedArray<Number>::n_array_elements;
  static const unsigned int dofs_per_cell = FEEvalCell::static_dofs_per_cell;

  void
  reinit(MF const &mf, CM &cm, AdditionalData const &ad,
         unsigned int level_mg_handler = numbers::invalid_unsigned_int) const;

  // TODO: remove
  virtual void reinit(const DoFHandler<dim> &, const Mapping<dim> &, void *,
                      const MGConstrainedDoFs & /*mg_constrained_dofs*/,
                      const unsigned int = numbers::invalid_unsigned_int) {
    AssertThrow(false, ExcMessage("OperatorBase::reinit to be removed!"));
  }

  void reinit_mf(const DoFHandler<dim> &dof_handler,
                 const Mapping<dim> &mapping,
                 MGConstrainedDoFs &mg_constrained_dofs, AdditionalData &ad,
                 const unsigned int level);

  /*
   * matrix vector multiplication
   */
  void apply(VNumber &dst, VNumber const &src) const;
  virtual void apply_add(VNumber &dst, VNumber const &src) const;

  void vmult(VNumber &dst, VNumber const &src) const;
  void vmult_add(VNumber &dst, VNumber const &src) const;

  void vmult_interface_down(VNumber &dst, VNumber const &src) const;
  void vmult_add_interface_up(VNumber &dst, VNumber const &src) const;

  /*
   *
   */
  void rhs(VNumber &dst) const { rhs(dst, 0.0); }
  void rhs(VNumber &dst, Number const time) const;
  // TODO: remove
  void rhs_add(VNumber &dst) const { rhs_add(dst, 0.0); }
  void rhs_add(VNumber &dst, Number const time) const;

  void evaluate(VNumber &dst, VNumber const &src, Number const time) const;
  void evaluate_add(VNumber &dst, VNumber const &src, Number const time) const;

  /*
   * point Jacobi method
   */
  void calculate_diagonal(VNumber &diagonal) const;
  void add_diagonal(VNumber &diagonal) const;
  void calculate_inverse_diagonal(VNumber &diagonal) const;

  /*
   * block Jacobi methods
   */
  void apply_block_jacobi(VNumber &dst, VNumber const &src) const;
  void apply_block_jacobi_add(VNumber &dst, VNumber const &src) const;

  // TODO: add matrix-free and block matrix version
  void apply_block_diagonal(VNumber &dst, VNumber const &src) const;
  void update_block_jacobi() const;
  virtual void add_block_jacobi_matrices(BMatrix &matrices) const;

  /*
   * sparse matrix (Trilinos) methods
   */
  void init_system_matrix(SMatrix &system_matrix, MPI_Comm comm) const;

  void calculate_system_matrix(SMatrix &system_matrix) const;

  /*
   * utility functions
   */
  types::global_dof_index m() const;

  types::global_dof_index n() const;

  Number el(const unsigned int, const unsigned int) const;

  bool is_empty_locally() const;

  const MF &get_data() const;

  unsigned int get_dof_index() const;

  unsigned int get_quad_index() const;

  AdditionalData const &get_operator_data() const;

  void initialize_dof_vector(VNumber &vector) const;

  void set_evaluation_time(double const evaluation_time_in);

  double get_evaluation_time() const;

protected:
  /*
   * methods to be overwritten
   */
  virtual void do_cell_integral(FEEvalCell &phi) const = 0;
  virtual void do_face_integral(FEEvalFace &p_n, FEEvalFace &p_p) const = 0;
  virtual void do_face_int_integral(FEEvalFace &p_n, FEEvalFace &p_p) const = 0;
  virtual void do_face_ext_integral(FEEvalFace &p_n, FEEvalFace &p_p) const = 0;
  virtual void
  do_boundary_integral(FEEvalFace &fe_eval, OperatorType const &operator_type,
                       types::boundary_id const &boundary_id) const = 0;

  /*
   * helper functions
   */
  void create_standard_basis(unsigned int j, FEEvalCell &phi) const;

  void create_standard_basis(unsigned int j, FEEvalFace &phi) const;

  void create_standard_basis(unsigned int j, FEEvalFace &phi1,
                             FEEvalFace &phi2) const;

  /*
   * functions to be called from matrix-free loops and cell_loops: vmult
   */
  void local_apply_cell(const MF & /*data*/, VNumber &dst, const VNumber &src,
                        const Range &range) const;

  void local_apply_face(const MF & /*data*/, VNumber &dst, const VNumber &src,
                        const Range &range) const;

  // homogenous
  void local_apply_boundary(const MF & /*data*/, VNumber & /*dst*/,
                            const VNumber & /*src*/,
                            const Range & /*range*/) const;

  void local_apply_inhom_cell(const MF & /*data*/, VNumber &dst,
                              const VNumber &src, const Range &range) const;

  void local_apply_inhom_face(const MF & /*data*/, VNumber &dst,
                              const VNumber &src, const Range &range) const;

  void local_apply_inhom_boundary(const MF & /*data*/, VNumber & /*dst*/,
                                  const VNumber & /*src*/,
                                  const Range & /*range*/) const;

  void local_apply_full_boundary(const MF & /*data*/, VNumber & /*dst*/,
                                 const VNumber & /*src*/,
                                 const Range & /*range*/) const;

  /*
   * ... diagonal
   */
  void local_apply_cell_diagonal(const MF & /*data*/, VNumber &dst,
                                 const VNumber & /*src*/,
                                 const Range &range) const;

  void local_apply_face_diagonal(const MF & /*data*/, VNumber &dst,
                                 const VNumber & /*src*/,
                                 const Range &range) const;

  void local_apply_boundary_diagonal(const MF & /*data*/, VNumber &dst,
                                     const VNumber & /*src*/,
                                     const Range &range) const;

  /*
   * ... block diagonal
   */
  void local_apply_block_diagonal(const MF & /*data*/, VNumber &dst,
                                  const VNumber &src, const Range &range) const;

  void
  cell_loop_apply_inverse_block_jacobi_matrices(const MF &data, VNumber &dst,
                                                const VNumber &src,
                                                const Range &cell_range) const;
  void local_apply_cell_block_diagonal(const MF & /*data*/, BMatrix &dst,
                                       const BMatrix & /*src*/,
                                       const Range &range) const;

  void local_apply_face_block_diagonal(const MF & /*data*/, BMatrix &dst,
                                       const BMatrix & /*src*/,
                                       const Range &range) const;

  void local_apply_boundary_block_diagonal(const MF & /*data*/, BMatrix &dst,
                                           const BMatrix & /*src*/,
                                           const Range &range) const;

  /*
   * ... sparse matrix
   */
  void local_apply_cell_system_matrix(const MF & /*data*/, SMatrix &dst,
                                      const SMatrix & /*src*/,
                                      const Range &range) const;

  void local_apply_face_system_matrix(const MF & /*data*/, SMatrix &dst,
                                      const SMatrix & /*src*/,
                                      const Range &range) const;

  void local_apply_boundary_system_matrix(const MF & /*data*/,
                                          SMatrix & /*dst*/,
                                          const SMatrix & /*src*/,
                                          const Range & /*range*/) const;

protected:
  mutable AdditionalData ad;
private:
  const bool do_eval_faces;
protected:
  mutable LazyWrapper<MF const> data;
private:
  mutable LazyWrapper<ConstraintMatrix> constraint;
  mutable bool is_dg;
  mutable bool is_mg;
  mutable unsigned int level_mg_handler;

  mutable std::vector<LAPACKFullMatrix<Number>> matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

protected:
  mutable double eval_time;
};

#include "operation_base.cpp"

#endif
