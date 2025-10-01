/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_OPERATORS_SOLUTION_TRANSFER_H_
#define EXADG_OPERATORS_SOLUTION_TRANSFER_H_

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/solution_transfer.h>

namespace ExaDG
{
template<int dim, typename VectorType>
class SolutionTransfer
{
public:
  /*
   * Constructor.
   */
  SolutionTransfer(dealii::DoFHandler<dim> const & dof_handler_in)
  {
    dof_handler = &dof_handler_in;
  }

  void
  prepare_coarsening_and_refinement(std::vector<VectorType *> & vectors)
  {
    std::vector<VectorType const *> vectors_old_grid_ptr(vectors.size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      vectors[i]->update_ghost_values();
      vectors_old_grid_ptr[i] = vectors[i];
    }

    pd_solution_transfer =
      std::make_shared<dealii::SolutionTransfer<dim, VectorType>>(*dof_handler);

    pd_solution_transfer->prepare_for_coarsening_and_refinement(vectors_old_grid_ptr);
  }

  void
  interpolate_after_coarsening_and_refinement(std::vector<VectorType *> & vectors)
  {
    // Note that the sequence of vectors per DofHandler/SolutionTransfer
    // defined in Operator<dim, Number>::prepare_coarsening_and_refinement()
    // and solution transfer calls here *need to match*.
    pd_solution_transfer->interpolate(vectors);
  }

private:
  std::shared_ptr<dealii::SolutionTransfer<dim, VectorType>> pd_solution_transfer;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_SOLUTION_TRANSFER_H_ */
