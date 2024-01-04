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
 *
 * Source: https://github.com/MeltPoolDG/MeltPoolDG
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 */

#ifndef INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_
#define INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_

// deal.II
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// ExaDG
#include <exadg/utilities/print_functions.h>


namespace ExaDG
{
struct AdaptiveMeshRefinementData
{
  AdaptiveMeshRefinementData()
    : trigger_every_n_time_steps(1),
      maximum_refinement_level(10),
      minimum_refinement_level(0),
      preserve_boundary_cells(false),
      fraction_of_cells_to_be_refined(0.0),
      fraction_of_cells_to_be_coarsened(0.0)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Enable adaptivity", true);
    print_parameter(pcout, "Triggered every n time steps", trigger_every_n_time_steps);
    print_parameter(pcout, "Maximum refinement level", maximum_refinement_level);
    print_parameter(pcout, "Minimum refinement level", minimum_refinement_level);
    print_parameter(pcout, "Preserve boundary cells", preserve_boundary_cells);
    print_parameter(pcout, "Fraction of cells to be refined", fraction_of_cells_to_be_refined);
    print_parameter(pcout, "Fraction of cells to be coarsened", fraction_of_cells_to_be_coarsened);
  }

  unsigned int trigger_every_n_time_steps;

  int  maximum_refinement_level;
  int  minimum_refinement_level;
  bool preserve_boundary_cells;

  double fraction_of_cells_to_be_refined;
  double fraction_of_cells_to_be_coarsened;
};

/**
 * Check if adaptive mesh refinement should be triggered depending on the time_step_number.
 */
inline bool
trigger_coarsening_and_refinement_now(unsigned int const trigger_every_n_time_steps,
                                      unsigned int const time_step_number)
{
  return ((time_step_number == 0) or not(time_step_number % trigger_every_n_time_steps));
}

/**
 * Limit the coarsening and refinement by adapting the flags set in the triangulation.
 * This considers the maximum and minimum refinement levels provided and might remove
 * refinement flags set on the boundary if requested.
 */
template<int dim>
void
limit_coarsening_and_refinement(dealii::Triangulation<dim> &       tria,
                                AdaptiveMeshRefinementData const & amr_data)
{
  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      // Clear refinement flags on maximum refinement level.
      if(cell->level() == amr_data.maximum_refinement_level)
      {
        cell->clear_refine_flag();
      }

      // Clear coarsening flags on minimum refinement level.
      if(cell->level() <= amr_data.minimum_refinement_level)
      {
        cell->clear_coarsen_flag();
      }

      // Do not coarsen/refine cells along boundary if requested
      if(amr_data.preserve_boundary_cells and cell->at_boundary())
      {
        cell->clear_refine_flag();
        cell->clear_coarsen_flag();
      }
    }
  }
}

/**
 * Loop over cells and check for refinement/coarsening flags set.
 */
template<int dim>
bool
any_cells_flagged_for_coarsening_or_refinement(dealii::Triangulation<dim> const & tria)
{
  bool any_flag_set = false;

  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      if(cell->refine_flag_set() or cell->coarsen_flag_set())
      {
        any_flag_set = true;
        break;
      }
    }
  }

  any_flag_set = dealii::Utilities::MPI::logical_or(any_flag_set, tria.get_communicator());

  return any_flag_set;
}

template<int dim, typename Number, typename VectorType>
void
mark_cells_kelly_error_estimator(dealii::Triangulation<dim> &              tria,
                                 dealii::DoFHandler<dim> const &           dof_handler,
                                 dealii::AffineConstraints<Number> const & constraints,
                                 dealii::Mapping<dim> const &              mapping,
                                 VectorType const &                        solution,
                                 unsigned int const                        n_face_quadrature_points,
                                 AdaptiveMeshRefinementData const &        amr_data)
{
  VectorType locally_relevant_solution;
  locally_relevant_solution.reinit(dof_handler.locally_owned_dofs(),
                                   dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                   dof_handler.get_communicator());
  locally_relevant_solution.copy_locally_owned_data_from(solution);
  constraints.distribute(locally_relevant_solution);
  locally_relevant_solution.update_ghost_values();

  dealii::QGauss<dim - 1> face_quadrature(n_face_quadrature_points);

  dealii::Vector<float> estimated_error_per_cell(tria.n_active_cells());

  dealii::KellyErrorEstimator<dim>::estimate(mapping,
                                             dof_handler,
                                             face_quadrature,
                                             {}, // Neumann BC
                                             locally_relevant_solution,
                                             estimated_error_per_cell);

  dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    tria,
    estimated_error_per_cell,
    amr_data.fraction_of_cells_to_be_refined,
    amr_data.fraction_of_cells_to_be_coarsened);
}


} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_ */
