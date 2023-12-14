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

#ifndef INCLUDE_EXADG_GRID_GRID_H_
#define INCLUDE_EXADG_GRID_GRID_H_

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
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> coarse_triangulations;

  /**
   * A vector of dealii::GridTools::PeriodicFacePair's for the coarse triangulations required for
   * h-multigrid with geometric coarsening types that require a vector of triangulations.
   *
   * This vector only contains levels coarser than the fine triangulation. The first entry
   * corresponds to the coarsest triangulation.
   */
  std::vector<PeriodicFacePairs> coarse_periodic_face_pairs;
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
    : mapping_fine_level(mapping), mapping_coarse_levels(mapping_coarse_levels)
  {
  }

  /**
   * Use this constructor if the fine-level mapping is of type ExaDG::MappingDoFVector.
   */
  MultigridMappings(std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector)
    : mapping_dof_vector_fine_level(mapping_dof_vector)
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
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
