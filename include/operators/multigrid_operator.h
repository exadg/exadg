/*
 * multigrid_operator.h
 *
 *  Created on: May 21, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_MULTIGRID_OPERATOR_H_
#define INCLUDE_OPERATORS_MULTIGRID_OPERATOR_H_

#include "multigrid_operator_base.h"

using namespace dealii;

template<int dim, typename Number, typename Operator>
class MultigridOperator : public MultigridOperatorBase<dim, Number>
{
public:
  typedef MultigridOperatorBase<dim, Number> Base;
  typedef typename Base::value_type          value_type;
  typedef typename Base::VectorType          VectorType;

  MultigridOperator(std::shared_ptr<Operator> op) : pde_operator(op)
  {
  }

  virtual ~MultigridOperator()
  {
  }

  std::shared_ptr<Operator>
  get_pde_operator() const
  {
    return pde_operator;
  }

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const
  {
    return pde_operator->get_constraint_matrix();
  }

  virtual const MatrixFree<dim, Number> &
  get_data() const
  {
    return pde_operator->get_data();
  }

  virtual unsigned int
  get_dof_index() const
  {
    return pde_operator->get_dof_index();
  }

  virtual types::global_dof_index
  m() const
  {
    return pde_operator->m();
  }

  virtual types::global_dof_index
  n() const
  {
    return pde_operator->n();
  }

  virtual Number
  el(const unsigned int i, const unsigned int j) const
  {
    return pde_operator->el(i, j);
  }

  virtual void
  initialize_dof_vector(VectorType & vector) const
  {
    pde_operator->initialize_dof_vector(vector);
  }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult(dst, src);
  }

  virtual void
  vmult_add(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult_add(dst, src);
  }

  virtual void
  vmult_interface_down(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult_interface_down(dst, src);
  }

  virtual void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult_add_interface_up(dst, src);
  }

  virtual void
  calculate_inverse_diagonal(VectorType & inverse_diagonal_entries) const
  {
    pde_operator->calculate_inverse_diagonal(inverse_diagonal_entries);
  }

  virtual void
  update_block_diagonal_preconditioner() const
  {
    pde_operator->update_block_diagonal_preconditioner();
  }

  virtual void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    pde_operator->apply_inverse_block_diagonal(dst, src);
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const
  {
    pde_operator->init_system_matrix(system_matrix);
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const
  {
    pde_operator->calculate_system_matrix(system_matrix);
  }
#endif

private:
  std::shared_ptr<Operator> pde_operator;
};



#endif /* INCLUDE_OPERATORS_MULTIGRID_OPERATOR_H_ */
