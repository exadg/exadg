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

#ifndef EXADG_MG_TRANSFER_GLOBAL_COARSENING_H
#define EXADG_MG_TRANSFER_GLOBAL_COARSENING_H

// deal.II
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/levels_hybrid_multigrid.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer.h>

namespace ExaDG
{
template<int dim, typename Number, typename VectorType>
class MGTransferGlobalCoarsening : virtual public MGTransfer<VectorType>
{
public:
  virtual ~MGTransferGlobalCoarsening()
  {
  }

  void
  reinit(dealii::MGLevelObject<std::shared_ptr<dealii::MatrixFree<dim, Number>>> & mg_matrixfree,
         unsigned int const dof_handler_index = 0);

  void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const final;

  void
  restrict_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const final;

  void
  prolongate_and_add(unsigned int const level,
                     VectorType &       dst,
                     VectorType const & src) const final;

private:
  dealii::MGLevelObject<dealii::MGTwoLevelTransfer<dim, VectorType>> transfers;

  std::unique_ptr<dealii::MGTransferGlobalCoarsening<dim, VectorType>>
    mg_transfer_global_coarsening;
};
} // namespace ExaDG

#endif // EXADG_MG_TRANSFER_GLOBAL_COARSENING_H
