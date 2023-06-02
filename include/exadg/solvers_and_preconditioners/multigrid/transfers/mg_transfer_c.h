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

#ifndef DG_CG_TRANSFER
#define DG_CG_TRANSFER

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer.h>

namespace ExaDG
{
template<int dim,
         typename Number,
         typename VectorType = dealii::LinearAlgebra::distributed::Vector<Number>,
         int components      = 1>
class MGTransferC : virtual public MGTransfer<VectorType>
{
public:
  MGTransferC(dealii::Mapping<dim> const &              mapping,
              dealii::MatrixFree<dim, Number> const &   matrixfree_dg,
              dealii::MatrixFree<dim, Number> const &   matrixfree_cg,
              dealii::AffineConstraints<Number> const & constraints_dg,
              dealii::AffineConstraints<Number> const & constraints_cg,
              unsigned int const                        level,
              unsigned int const                        fe_degree,
              unsigned int const                        dof_handler_index = 0);

  virtual ~MGTransferC();

  void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const final;

  void
  restrict_and_add(unsigned int const /*level*/,
                   VectorType &       dst,
                   VectorType const & src) const final;

  void
  prolongate_and_add(unsigned int const /*level*/,
                     VectorType &       dst,
                     VectorType const & src) const final;

private:
  template<int degree>
  void
  do_interpolate(VectorType & dst, VectorType const & src) const;

  template<int degree>
  void
  do_restrict_and_add(VectorType & dst, VectorType const & src) const;

  template<int degree>
  void
  do_prolongate_and_add(VectorType & dst, VectorType const & src) const;

  unsigned int const              fe_degree;
  dealii::MatrixFree<dim, Number> data_composite;
};

} // namespace ExaDG

#endif
