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

#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

// deal.II
#include <deal.II/base/mg_level_object.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer.h>

namespace ExaDG
{
template<int dim,
         typename Number,
         typename VectorType = dealii::LinearAlgebra::distributed::Vector<Number>>
class MGTransferGlobalRefinement : virtual public MGTransfer<VectorType>
{
public:
  virtual ~MGTransferGlobalRefinement()
  {
  }

  void
  reinit(dealii::Mapping<dim> const &                                              mapping,
         dealii::MGLevelObject<std::shared_ptr<dealii::MatrixFree<dim, Number>>> & mg_matrixfree,
         dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>> & mg_constrained_dofs,
         unsigned int const                                                  dof_handler_index = 0);

  virtual void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const final;

  virtual void
  restrict_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const final;

  virtual void
  prolongate_and_add(unsigned int const level,
                     VectorType &       dst,
                     VectorType const & src) const final;

private:
  dealii::MGLevelObject<std::shared_ptr<MGTransfer<VectorType>>> mg_level_object;
};

} // namespace ExaDG

#endif
