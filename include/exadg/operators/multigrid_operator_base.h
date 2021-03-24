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

#ifndef OPERATOR_PRECONDITIONABLE_H
#define OPERATOR_PRECONDITIONABLE_H

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MultigridOperatorBase : public dealii::Subscriptor
{
public:
  typedef Number                                     value_type;
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MultigridOperatorBase() : dealii::Subscriptor()
  {
  }

  virtual ~MultigridOperatorBase()
  {
  }

  virtual AffineConstraints<Number> const &
  get_affine_constraints() const = 0;

  virtual MatrixFree<dim, Number> const &
  get_matrix_free() const = 0;

  virtual unsigned int
  get_dof_index() const = 0;

  virtual types::global_dof_index
  m() const = 0;

  virtual types::global_dof_index
  n() const = 0;

  virtual Number
  el(unsigned int const, const unsigned int) const = 0;

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

#ifdef DEAL_II_WITH_PETSC
  virtual void
  init_system_matrix(PETScWrappers::MPI::SparseMatrix & system_matrix) const = 0;

  virtual void
  calculate_system_matrix(PETScWrappers::MPI::SparseMatrix & system_matrix) const = 0;
#endif
};

} // namespace ExaDG

#endif
