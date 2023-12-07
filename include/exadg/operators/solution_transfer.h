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

#ifndef INCLUDE_EXADG_OPERATORS_SOLUTION_TRANSFER_H
#define INCLUDE_EXADG_OPERATORS_SOLUTION_TRANSFER_H

// deal.II
#include <deal.II/distributed/solution_transfer.h>
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
    // Container vectors_old_grid hold vectors for interpolation *REQUIRED* for
    // interpolate_after_coarsening_and_refinement(). Thus makes an actual copy.
    // In the case of parallel::distributed::Triangulation, pointers are sufficient.
    vectors_old_grid.resize(vectors.size());
    std::vector<VectorType const *> vectors_old_grid_ptr(vectors.size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      vectors[i]->update_ghost_values();

      if(is_parallel_distributed_triangulation())
      {
        vectors_old_grid_ptr[i] = vectors[i];
      }
      else
      {
        VectorType const & vector = *vectors[i];
        dealii::IndexSet   indices(vector.size());
        indices.add_range(0, vector.size());
        vectors_old_grid[i].reinit(vector.locally_owned_elements(),
                                   indices,
                                   vector.get_mpi_communicator());
        vectors_old_grid[i].copy_locally_owned_data_from(vector);
        vectors_old_grid[i].update_ghost_values();
      }
    }

    // SolutionTransfer object type depends on the triangulation type.
    if(is_parallel_distributed_triangulation())
    {
      pd_solution_transfer =
        std::make_shared<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>(
          *dof_handler);

      pd_solution_transfer->prepare_for_coarsening_and_refinement(vectors_old_grid_ptr);
    }
    else
    {
      solution_transfer = std::make_shared<dealii::SolutionTransfer<dim, VectorType>>(*dof_handler);
      solution_transfer->prepare_for_coarsening_and_refinement(vectors_old_grid);
    }
  }

  void
  interpolate_after_coarsening_and_refinement(std::vector<VectorType *> & vectors)
  {
    // Note that the sequence of vectors per DofHandler/SolutionTransfer
    // defined in Operator<dim, Number>::prepare_coarsening_and_refinement()
    // and solution transfer calls here *need to match*.
    if(is_parallel_distributed_triangulation())
    {
      pd_solution_transfer->interpolate(vectors);
    }
    else
    {
      // Initialize ghosted vectors for interpolation.
      std::vector<VectorType> vectors_new_grid(vectors.size());
      for(unsigned int i = 0; i < vectors.size(); ++i)
      {
        VectorType const & vector = *vectors[i];
        dealii::IndexSet   indices(vector.size());
        indices.add_range(0, vector.size());
        vectors_new_grid[i].reinit(vector.locally_owned_elements(),
                                   indices,
                                   vector.get_mpi_communicator());
      }

      solution_transfer->interpolate(vectors_old_grid, vectors_new_grid);

      // Copy ghosted vectors to output.
      for(unsigned int i = 0; i < vectors.size(); ++i)
      {
        vectors[i]->copy_locally_owned_data_from(vectors_new_grid[i]);
      }
    }

    // Clear the containers holding actual copies or pointers depending
    // on the SolutionTransfer type. In both cases, the containers need
    // to be in scope when execute_coarsening_and_refinement() is called.
    vectors_old_grid.clear();
  }

private:
  bool
  is_parallel_distributed_triangulation() const
  {
    return (dynamic_cast<dealii::parallel::distributed::Triangulation<dim> const *>(
      &dof_handler->get_triangulation()));
  }

  std::vector<VectorType> vectors_old_grid;

  std::shared_ptr<dealii::SolutionTransfer<dim, VectorType>> solution_transfer;
  std::shared_ptr<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>
    pd_solution_transfer;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler;
};
} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_SOLUTION_TRANSFER_H */
