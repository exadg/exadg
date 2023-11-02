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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
template<int dim, typename Number>
class MultigridOperatorBase : public dealii::Subscriptor
{
public:
  typedef Number                                             value_type;
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  MultigridOperatorBase() : dealii::Subscriptor()
  {
  }

  virtual ~MultigridOperatorBase()
  {
  }

  virtual dealii::AffineConstraints<Number> const &
  get_affine_constraints() const = 0;

  virtual dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const = 0;

  virtual unsigned int
  get_dof_index() const = 0;

  virtual dealii::types::global_dof_index
  m() const = 0;

  virtual dealii::types::global_dof_index
  n() const = 0;

  virtual Number
  el(unsigned int const, unsigned int const) const = 0;

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

  virtual void
  apply_inverse_additive_schwarz_matrices(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  compute_factorized_additive_schwarz_matrices() const = 0;

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix,
                     MPI_Comm const &                         mpi_comm) const = 0;

  virtual void
  calculate_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const = 0;
#endif

#ifdef DEAL_II_WITH_PETSC
  virtual void
  init_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
                     MPI_Comm const &                           mpi_comm) const = 0;

  virtual void
  calculate_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const = 0;
#endif
};

} // namespace ExaDG

#endif
