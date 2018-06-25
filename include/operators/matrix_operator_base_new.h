#ifndef MATRIX_OPERATOR_BASE_NEW
#define MATRIX_OPERATOR_BASE_NEW

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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

#include "matrix_operator_base.h"

using namespace dealii;

template <int dim, typename Number = double>
class MatrixOperatorBaseNew : public MatrixOperatorBase {

public:
  typedef Number value_type;
  static const int DIM = dim;

  virtual void clear();

  virtual void vmult(parallel::distributed::Vector<Number> &dst,
                     const parallel::distributed::Vector<Number> &src) const;

  virtual void Tvmult(parallel::distributed::Vector<Number> &dst,
                      const parallel::distributed::Vector<Number> &src) const;

  virtual void
  Tvmult_add(parallel::distributed::Vector<Number> &dst,
             const parallel::distributed::Vector<Number> &src) const;

  virtual void
  vmult_add(parallel::distributed::Vector<Number> &dst,
            const parallel::distributed::Vector<Number> &src) const;

  virtual void
  vmult_interface_down(parallel::distributed::Vector<Number> &dst,
                       const parallel::distributed::Vector<Number> &src) const;

  virtual void vmult_add_interface_up(
      parallel::distributed::Vector<Number> &dst,
      const parallel::distributed::Vector<Number> &src) const;

  virtual void rhs(parallel::distributed::Vector<Number> &dst) const;

  virtual void rhs_add(parallel::distributed::Vector<Number> &dst) const;

  virtual void
  apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const;

  virtual void disable_mean_value_constraint();

  virtual types::global_dof_index m() const;

  virtual types::global_dof_index n() const;

  virtual Number el(const unsigned int, const unsigned int) const;

  virtual void
  initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const;

  virtual void calculate_inverse_diagonal(
      parallel::distributed::Vector<Number> &inverse_diagonal_entries) const;

  virtual void
  apply_block_jacobi(parallel::distributed::Vector<Number> &dst,
                     parallel::distributed::Vector<Number> const &src) const;

  virtual void update_block_jacobi() const;

  virtual const AlignedVector<VectorizedArray<Number>> &
  get_array_penalty_parameter() const;

  virtual double get_penalty_factor() const;

  virtual bool is_empty_locally() const;

  virtual const MatrixFree<dim, Number> &get_data() const;

  virtual unsigned int get_dof_index() const;

  virtual void cell(MeshWorker::DoFInfo<dim, dim> &dinfo,
                    typename MeshWorker::IntegrationInfo<dim> &info) const;

  virtual void boundary(MeshWorker::DoFInfo<dim, dim> &dinfo,
                        typename MeshWorker::IntegrationInfo<dim> &info) const;

  virtual void face(MeshWorker::DoFInfo<dim, dim> &dinfo1,
                    MeshWorker::DoFInfo<dim, dim> &dinfo2,
                    typename MeshWorker::IntegrationInfo<dim> &info1,
                    typename MeshWorker::IntegrationInfo<dim> &info2) const;

  virtual const ConstraintMatrix &get_constraint_matrix() const;
  
  virtual MatrixOperatorBaseNew<dim, Number>* get_new(unsigned int deg) const;
  
  
  virtual void reinit (const DoFHandler<dim>          &dof_handler,
               const Mapping<dim>             &mapping,
               //const LaplaceOperatorData<dim> &operator_data,
               void * operator_data,
               const MGConstrainedDoFs        &mg_constrained_dofs,
               const unsigned int             level = numbers::invalid_unsigned_int);
};

#endif