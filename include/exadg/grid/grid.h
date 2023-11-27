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
 * This class stores pointers to fine and coarse mappings of type MappingDoFVector for use in
 * multigrid with h-transfer.
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
   * This function stores pointers to fine and coarse mappings of type MappingDoFVector for use in
   * multigrid with h-transfer.
   *
   * The default implementation uses an interpolation of the fine-level mapping to coarser grids.
   * You can overwrite this function in order to realize a user-specific construction of the mapping
   * on coarser grids.
   *
   */
  std::function<void(
    std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
    std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)>
    initialize_coarse_mappings =
      [&](std::shared_ptr<dealii::Triangulation<dim> const> const & fine_triangulation,
          std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const &
            coarse_triangulations) {
        MappingTools::initialize_coarse_mappings<dim, Number>(mapping_dof_vector_coarse_levels,
                                                              mapping_dof_vector_fine_level,
                                                              fine_triangulation,
                                                              coarse_triangulations);
      };

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
