#ifndef OPERATOR_PRECONDITIONABLE_H
#define OPERATOR_PRECONDITIONABLE_H

#include <deal.II/matrix_free/matrix_free.h>

#include "linear_operator_base.h"

template<int dim, typename Number>
class MultigridOperatorBase : public LinearOperatorBase
{
public:
  typedef Number                                     value_type;
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~MultigridOperatorBase()
  {
  }

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const = 0;

  virtual const MatrixFree<dim, Number> &
  get_matrix_free() const = 0;

  virtual unsigned int
  get_dof_index() const = 0;

  virtual types::global_dof_index
  m() const = 0;

  virtual types::global_dof_index
  n() const = 0;

  virtual Number
  el(const unsigned int, const unsigned int) const = 0;

  virtual void
  initialize_dof_vector(VectorType & vector) const = 0;

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_interface_down(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  calculate_inverse_diagonal(VectorType & inverse_diagonal_entries) const = 0;

  virtual void
  update_block_diagonal_preconditioner() const = 0;

  virtual void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const = 0;

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const = 0;

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const = 0;
#endif
};

#endif
