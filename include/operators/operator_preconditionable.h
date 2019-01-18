#ifndef OPERATOR_PRECONDITIONABLE_H
#define OPERATOR_PRECONDITIONABLE_H

#include <deal.II/matrix_free/matrix_free.h>

#include "linear_operator_base_new.h"

template<int dim>
struct PreconditionableOperatorData
{
public:
  virtual bool
  do_use_cell_based_loops() const
  {
    AssertThrow(false, ExcMessage("Not implemented yet!"));
    return true;
  }

  virtual void
  set_dof_index(const int dof_index)
  {
    (void)dof_index;
    AssertThrow(false, ExcMessage("Not implemented yet!"));
  }

  virtual void
  set_quad_index(const int quad_index)
  {
    (void)quad_index;
    AssertThrow(false, ExcMessage("Not implemented yet!"));
  }

  virtual UpdateFlags
  get_mapping_update_flags() const
  {
    AssertThrow(false, ExcMessage("Not implemented yet!"));
    return update_default;
  }

  virtual UpdateFlags
  get_mapping_update_flags_inner_faces() const
  {
    AssertThrow(false, ExcMessage("Not implemented yet!"));
    return update_default;
  }

  virtual UpdateFlags
  get_mapping_update_flags_boundary_faces() const
  {
    AssertThrow(false, ExcMessage("Not implemented yet!"));
    return update_default;
  }
};


template<int dim, typename Number>
class PreconditionableOperator : virtual public LinearOperatorBaseNew<Number>
{
public:
  typedef LinearOperatorBaseNew<Number> Parent;

  typedef typename Parent::value_type value_type;
  static const int                    DIM = dim;
  typedef typename Parent::VectorType VectorType;

  /*
   * Initialization
   */
  virtual void
  reinit_preconditionable_operator_data(
    MatrixFree<dim, Number> const &           matrix_free,
    AffineConstraints<double> const &         constraint_matrix,
    PreconditionableOperatorData<dim> const & operator_data_in) const = 0;

  /*
   *
   */
  virtual PreconditionableOperator<dim, Number> *
  get_new(unsigned int deg) const = 0;

  virtual bool
  is_singular() const = 0;

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const = 0;

  virtual const MatrixFree<dim, Number> &
  get_data() const = 0;

  virtual unsigned int
  get_dof_index() const = 0;

  /*
   * Actual preconditioning methods
   */
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