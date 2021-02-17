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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_

// deal.II
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

// ExaDG
#include <exadg/operators/multigrid_operator_base.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer.h>

namespace ExaDG
{
using namespace dealii;

/**
 * Specialized matrix-free implementation that overloads the copy_to_mg function for proper
 * initialization of the vectors in matrix-vector products.
 */
template<int dim, typename Number>
class MGTransferH : public MGTransferMatrixFree<dim, Number>,
                    public MGTransfer<LinearAlgebra::distributed::Vector<Number>>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MGTransferH(std::map<unsigned int, unsigned int> level_to_triangulation_level_map,
              const DoFHandler<dim> &              dof_handler);

  virtual ~MGTransferH()
  {
  }

  void
  set_operator(
    const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> & operator_in);

  virtual void
  prolongate(unsigned int const to_level, VectorType & dst, VectorType const & src) const;

  virtual void
  restrict_and_add(unsigned int const from_level, VectorType & dst, VectorType const & src) const;

  virtual void
  interpolate(unsigned int const level_in, VectorType & dst, VectorType const & src) const;

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  void
  copy_to_mg(const DoFHandler<dim, dim> & mg_dof,
             MGLevelObject<VectorType> &  dst,
             const VectorType &           src) const;

private:
  const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> * underlying_operator;

  /*
   * This map converts the multigrid level as used in the V-cycle to an actual level in the
   * triangulation (this is necessary since both numbers might not equal e.g. in the case of hybrid
   * multigrid involving p-transfer on the same triangulation level).
   */
  mutable std::map<unsigned int, unsigned int> level_to_triangulation_level_map;

  const DoFHandler<dim> & dof_handler;
};
} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_ */
