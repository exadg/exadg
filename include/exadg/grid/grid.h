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
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>

namespace ExaDG
{
template<int dim>
ElementType
get_element_type(dealii::Triangulation<dim> const & tria)
{
  if(tria.all_reference_cells_are_simplex())
  {
    return ElementType::Simplex;
  }
  else if(tria.all_reference_cells_are_hyper_cube())
  {
    return ElementType::Hypercube;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter element_type."));
    return ElementType::Hypercube;
  }
}

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
    PeriodicFaces;

  /**
   * Constructor.
   */
  Grid(GridData const & data, MPI_Comm const & mpi_comm);

  /**
   * dealii::Triangulation::MeshSmoothing
   */
  typename dealii::Triangulation<dim>::MeshSmoothing mesh_smoothing;

  /**
   * dealii::Triangulation.
   */
  std::shared_ptr<dealii::Triangulation<dim>> triangulation;

  /**
   * a vector of coarse triangulations required for global coarsening multigrid
   */
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> coarse_triangulations;

  /**
   * dealii::GridTools::PeriodicFacePair's.
   */
  PeriodicFaces periodic_faces;

  /**
   * dealii::Mapping.
   */
  std::shared_ptr<dealii::Mapping<dim>> mapping;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
