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

#ifndef EXADG_GRID_GRID_H_
#define EXADG_GRID_GRID_H_

// deal.II
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>

namespace ExaDG
{
/**
 * A struct of dealii data structures occurring in close proximity to each other so that it makes
 * sense to group them together to keep interfaces lean.
 */
template<int dim>
class Grid
{
public:
  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    PeriodicFacePairs;

  /**
   * dealii::Triangulation.
   */
  std::shared_ptr<dealii::Triangulation<dim>> triangulation;

  /**
   * dealii::GridTools::PeriodicFacePair's.
   */
  PeriodicFacePairs periodic_face_pairs;

  /**
   * A vector of coarse triangulations required for h-multigrid with geometric coarsening types that
   * require a vector of triangulations.
   *
   * This vector only contains levels coarser than the fine triangulation. The first entry
   * corresponds to the coarsest triangulation.
   */
  mutable std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> coarse_triangulations;

  /**
   * A vector of dealii::GridTools::PeriodicFacePair's for the coarse triangulations required for
   * h-multigrid with geometric coarsening types that require a vector of triangulations.
   *
   * This vector only contains levels coarser than the fine triangulation. The first entry
   * corresponds to the coarsest triangulation.
   */
  std::vector<PeriodicFacePairs> coarse_periodic_face_pairs;

  /**
   * Functionality to stash the `manifold_id`s of the `triangulation` and `coarse_triangulations`.
   */
  void
  stash_manifold_ids() const
  {
    // stash `manifold_id`s of fine triangulation
    if(has_non_flat_manifold_ids())
    {
      stashed_manifold_ids.resize(triangulation->n_active_cells());

      unsigned int counter = 0;
      for(auto const & cell : triangulation->active_cell_iterators())
        stashed_manifold_ids[counter++] = cell->manifold_id();

      // stash `manifold_id`s of coarse triangulations
      coarse_stashed_manifold_ids.resize(coarse_triangulations.size());
      for(unsigned int i = 0; i < coarse_triangulations.size(); ++i)
      {
        coarse_stashed_manifold_ids[i].resize(coarse_triangulations[i]->n_active_cells());

        counter = 0;
        for(auto const & cell : coarse_triangulations[i]->active_cell_iterators())
          coarse_stashed_manifold_ids[i][counter++] = cell->manifold_id();
      }
    }
  }

  /**
   * Set the `manifold_id`s to the default value of `manifold_id::flat`.
   */
  void
  set_manifold_ids_to_flat() const
  {
    if(has_non_flat_manifold_ids())
    {
      // set `manifold_id`s of fine triangulation to flat
      for(auto & cell : triangulation->active_cell_iterators())
        cell->set_manifold_id(dealii::numbers::flat_manifold_id);

      // set `manifold_id`s of coarse triangulations to flat
      for(unsigned int i = 0; i < coarse_triangulations.size(); ++i)
      {
        for(auto & cell : coarse_triangulations[i]->active_cell_iterators())
          cell->set_manifold_id(dealii::numbers::flat_manifold_id);
      }
    }
  }

  /**
   * Restore the `manifold_id`s of the triangulation.
   */
  void
  unstash_manifold_ids() const
  {
    // restore `manifold_id`s of fine triangulation
    {
      AssertThrow(stashed_manifold_ids.size() == triangulation->n_active_cells(),
                  dealii::ExcMessage("The number of stashed `manifold_id`s does not match the "
                                     "number of active cells on the fine triangulation."));

      unsigned int counter = 0;
      for(auto & cell : triangulation->active_cell_iterators())
        cell->set_manifold_id(stashed_manifold_ids[counter++]);

      // restore `manifold_id`s of coarse triangulations
      AssertThrow(coarse_stashed_manifold_ids.size() == coarse_triangulations.size(),
                  dealii::ExcMessage("The number of levels of stashed `manifold_id`s does not "
                                     "match the number of coarse triangulations."));

      for(unsigned int i = 0; i < coarse_triangulations.size(); ++i)
      {
        AssertThrow(coarse_stashed_manifold_ids[i].size() ==
                      coarse_triangulations[i]->n_active_cells(),
                    dealii::ExcMessage("The number of stashed `manifold_id`s does not match the "
                                       "number of active cells on the coarse triangulation."));

        counter = 0;
        for(auto & cell : coarse_triangulations[i]->active_cell_iterators())
          cell->set_manifold_id(coarse_stashed_manifold_ids[i][counter++]);
      }
    }
  }

  /**
   * Utility function to undo any refinement and return to the initial coarse mesh.
   * The coarsening history is stored to a bitvector to recover it at a later point.
   */
  void
  undo_refinement_fine_level() const
  {
    // Reset history member variable.
    mesh_history.clear();
    coarsening_loops_executed = 0;

    bool any_cell_coarsened = true;
    while(any_cell_coarsened)
    {
      any_cell_coarsened = false;

      std::cout << "Undoing refinement step..." << coarsening_loops_executed << std::endl;

      // Set coarsen flags in all locally owned cells if possible.
      for(auto const & cell : triangulation->active_cell_iterators())
      {
        if(cell->is_locally_owned())
        {
          cell->set_coarsen_flag();
        }
      }

      // Store coarsening flags to bitvector and coarsen mesh.
      coarsening_loops_executed++;
      triangulation->save_coarsen_flags(mesh_history);
      unsigned int n_active_cells_pre = triangulation->n_global_active_cells();
      triangulation->execute_coarsening_and_refinement();
      unsigned int n_active_cells_post = triangulation->n_global_active_cells();

      // Synchronize break criterion.
      if(n_active_cells_post < n_active_cells_pre)
      {
        any_cell_coarsened = true;
        std::cout << "found a cell to coarsen..." << std::endl;
      }
    }
  }

  /**
   * Utility function to redo any refinement and return to the initial fine mesh
   * based on the coarsening history read from the bitvector.
   */
  void
  redo_refinement_fine_level() const
  {
    AssertThrow(mesh_history.size() > 0,
                dealii::ExcMessage("No coarsening history stored to redo refinement."));

    for(unsigned int step = 0; step < coarsening_loops_executed; ++step)
    {
      std::cout << "Redoing refinement step..." << step << std::endl;
      triangulation->load_coarsen_flags(mesh_history);

      // If the coarsen flags were set before, we refine at this point.
      for(auto const & cell : triangulation->active_cell_iterators())
      {
        if(cell->is_locally_owned())
        {
          if(cell->coarsen_flag_set())
          {
            cell->clear_coarsen_flag();
            cell->set_refine_flag();
          }
        }
      }

      triangulation->execute_coarsening_and_refinement();
    }
  }

private:
  /**
   * Check if the triangulation has non-flat manifold ids.
   */
  bool
  has_non_flat_manifold_ids() const
  {
    std::vector<dealii::types::manifold_id> const manifold_ids = triangulation->get_manifold_ids();

    bool has_only_flat_manifold_id =
      manifold_ids.size() == 1 and manifold_ids[0] == dealii::numbers::flat_manifold_id;

    return not has_only_flat_manifold_id;
  }

  /**
   * Stashed `manifold_id`s of the triangulation.
   */
  mutable std::vector<dealii::types::manifold_id>              stashed_manifold_ids;
  mutable std::vector<std::vector<dealii::types::manifold_id>> coarse_stashed_manifold_ids;

  /**
   * Coarsening history of the mesh stored to a bitvector to recover it at a later point.
   */
  mutable unsigned int      coarsening_loops_executed;
  mutable std::vector<bool> mesh_history;
};

/**
 * This class handles mappings for use in multigrid.
 *
 * The lambda function initialize_coarse_mappings() can be overwritten in order to realize a
 * user-specific construction of the mapping on coarser grids.
 *
 */
template<int dim, typename Number>
class MultigridMappings
{
public:
  /**
   * Use this constructor if the fine-level mapping is a "normal" dealii::Mapping.
   */
  MultigridMappings(std::shared_ptr<dealii::Mapping<dim>> mapping,
                    std::shared_ptr<dealii::Mapping<dim>> mapping_coarse_levels)
    : mapping_fine_level(mapping),
      mapping_coarse_levels(mapping_coarse_levels),
      degree_coarse_mappings(1)
  {
  }

  /**
   * Use this constructor if the fine-level mapping is of type ExaDG::MappingDoFVector.
   */
  MultigridMappings(std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector,
                    unsigned int const                             degree_coarse_mappings)
    : mapping_dof_vector_fine_level(mapping_dof_vector),
      degree_coarse_mappings(degree_coarse_mappings)
  {
  }

  /**
   * Initializes the multigrid mappings on coarse h levels.
   */
  void
  initialize_coarse_mappings(Grid<dim> const & grid, unsigned int const n_h_levels)
  {
    // we need to initialize mappings for coarse levels only if we have a mapping of type
    // MappingDoFVector
    if(mapping_dof_vector_fine_level.get())
    {
      if(n_h_levels > 1)
      {
        mapping_dof_vector_coarse_levels.resize(n_h_levels - 1);

        lambda_initialize_coarse_mappings(grid.triangulation, grid.coarse_triangulations);
      }
    }
    else // standard dealii::Mapping
    {
      AssertThrow(mapping_fine_level.get(),
                  dealii::ExcMessage("Fine-level mapping is uninitialized."));

      // when using standard dealii::Mapping's, there is nothing to do, i.e. we assume that all the
      // mappings have been set up prior to calling the constructor of this class.
    }
  }

  /**
   * Returns the dealii::Mapping for a given h_level of n_h_levels.
   */
  dealii::Mapping<dim> const &
  get_mapping(unsigned int const h_level, unsigned int const n_h_levels) const
  {
    if(mapping_dof_vector_fine_level.get()) // ExaDG::MappingDoFVector
    {
      // fine level
      if(h_level == n_h_levels - 1)
      {
        return *(mapping_dof_vector_fine_level->get_mapping());
      }
      else // coarse levels
      {
        AssertThrow(h_level < mapping_dof_vector_coarse_levels.size(),
                    dealii::ExcMessage("Vector of coarse mappings seems to have incorrect size."));

        return *(mapping_dof_vector_coarse_levels[h_level]->get_mapping());
      }
    }
    else // standard dealii::Mapping
    {
      // use mapping_fine_level on the fine level or on all levels if mapping_coarse_levels is
      // uninitialized
      if(h_level == n_h_levels - 1 or not(mapping_coarse_levels.get()))
      {
        AssertThrow(mapping_fine_level.get(),
                    dealii::ExcMessage("Fine-level mapping is uninitialized."));

        return *mapping_fine_level;
      }
      else // coarse levels
      {
        AssertThrow(mapping_coarse_levels.get(),
                    dealii::ExcMessage("mapping_coarse_levels is uninitialized."));

        return *mapping_coarse_levels;
      }
    }
  }

  /**
   * This function creates coarse mappings of type MappingDoFVector for use in
   * multigrid with h-transfer given a fine level mapping of type MappingDoFVector.
   *
   * The default implementation uses an interpolation of the fine-level mapping to coarser grids.
   * You can overwrite this function in order to realize a user-specific construction of the mapping
   * on coarser grids.
   *
   */
  std::function<void(
    std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
    std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)>
    lambda_initialize_coarse_mappings =
      [&](std::shared_ptr<dealii::Triangulation<dim> const> const & fine_triangulation,
          std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const &
            coarse_triangulations) {
        AssertThrow(
          mapping_dof_vector_fine_level.get(),
          dealii::ExcMessage(
            "Coarse mappings can not be initialized because fine level mapping is invalid."));

        MappingTools::initialize_coarse_mappings<dim, Number>(mapping_dof_vector_coarse_levels,
                                                              degree_coarse_mappings,
                                                              mapping_dof_vector_fine_level,
                                                              fine_triangulation,
                                                              coarse_triangulations);
      };

private:
  /**
   * dealii::Mapping on fine level
   */
  std::shared_ptr<dealii::Mapping<dim>> mapping_fine_level;

  /**
   * dealii::Mapping on coarse levels. This shared pointer is allowed to be un-initialized when
   * calling the constructor of this class. In this case, mapping_fine_level is taken as the mapping
   * for all multigrid h-levels.
   */
  std::shared_ptr<dealii::Mapping<dim>> mapping_coarse_levels;

  /**
   * MappingDoFVector object on fine triangulation level.
   */
  std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector_fine_level;

  /**
   * Vector of coarse mappings of type MappingDoFVector for all h-multigrid levels coarser than
   * the fine triangulation. The first entry corresponds to the coarsest triangulation.
   */
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> mapping_dof_vector_coarse_levels;

  /**
   * Degree for coarse-grid mappings. This variable is only relevant for mappigns of type
   * MappingDoFVector.
   */
  unsigned int const degree_coarse_mappings;
};

} // namespace ExaDG

#endif /* EXADG_GRID_GRID_H_ */
