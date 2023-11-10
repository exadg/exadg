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

#ifndef INCLUDE_EXADG_OPERATORS_MULTIGRID_OPERATOR_H_
#define INCLUDE_EXADG_OPERATORS_MULTIGRID_OPERATOR_H_

#include <exadg/operators/multigrid_operator_base.h>

namespace ExaDG
{
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
    AssertThrow(pde_operator.get() != 0, dealii::ExcMessage("Invalid pointer"));

    return pde_operator;
  }

  dealii::AffineConstraints<typename Operator::value_type> const &
  get_affine_constraints() const final
  {
    return pde_operator->get_affine_constraints();
  }

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const final
  {
    return pde_operator->get_matrix_free();
  }

  unsigned int
  get_dof_index() const final
  {
    return pde_operator->get_dof_index();
  }

  dealii::types::global_dof_index
  m() const final
  {
    return pde_operator->m();
  }

  dealii::types::global_dof_index
  n() const final
  {
    return pde_operator->n();
  }

  Number
  el(unsigned int const i, unsigned int const j) const final
  {
    return pde_operator->el(i, j);
  }

  void
  initialize_dof_vector(VectorType & vector) const final
  {
    pde_operator->initialize_dof_vector(vector);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->vmult(dst, src);
  }

  void
  vmult_add(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->vmult_add(dst, src);
  }

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->vmult_interface_down(dst, src);
  }

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->vmult_add_interface_up(dst, src);
  }

  void
  calculate_inverse_diagonal(VectorType & inverse_diagonal_entries) const final
  {
    pde_operator->calculate_inverse_diagonal(inverse_diagonal_entries);
  }

  void
  update_block_diagonal_preconditioner() const final
  {
    pde_operator->update_block_diagonal_preconditioner();
  }

  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->apply_inverse_block_diagonal(dst, src);
  }

  virtual void
  apply_inverse_additive_schwarz_matrices(VectorType & dst, VectorType const & src) const final
  {
    pde_operator->apply_inverse_additive_schwarz_matrices(dst, src);
  }

  virtual void
  compute_factorized_additive_schwarz_matrices() const final
  {
    pde_operator->compute_factorized_additive_schwarz_matrices();
  }

#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix,
                     MPI_Comm const &                         mpi_comm) const final
  {
    pde_operator->init_system_matrix(system_matrix, mpi_comm);
  }

  void
  calculate_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const final
  {
    pde_operator->calculate_system_matrix(system_matrix);
  }
#endif

#ifdef DEAL_II_WITH_PETSC
  void
  init_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
                     MPI_Comm const &                           mpi_comm) const final
  {
    pde_operator->init_system_matrix(system_matrix, mpi_comm);
  }

  void
  calculate_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const final
  {
    pde_operator->calculate_system_matrix(system_matrix);
  }
#endif

private:
  std::shared_ptr<Operator> pde_operator;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_MULTIGRID_OPERATOR_H_ */
