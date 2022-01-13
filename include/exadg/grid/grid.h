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
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_motion_interface.h>

namespace ExaDG
{
using namespace dealii;

struct GridData
{
  GridData()
    : triangulation_type(TriangulationType::Distributed),
      n_refine_global(0),
      n_subdivisions_1d_hypercube(1),
      mapping_degree(1)
  {
  }

  TriangulationType triangulation_type;

  unsigned int n_refine_global;

  // only relevant for hypercube geometry/mesh
  unsigned int n_subdivisions_1d_hypercube;

  unsigned int mapping_degree;

  // TODO: path to a grid file
  // std::string grid_file;
};

template<int dim, typename Number>
class Grid
{
public:
  typedef
    typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      PeriodicFaces;

  /**
   * Constructor.
   */
  Grid(GridData const & data, MPI_Comm const & mpi_comm)
  {
    // triangulation
    if(data.triangulation_type == TriangulationType::Serial)
    {
      AssertDimension(Utilities::MPI::n_mpi_processes(mpi_comm), 1);
      triangulation = std::make_shared<Triangulation<dim>>();
    }
    else if(data.triangulation_type == TriangulationType::Distributed)
    {
      triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(
        mpi_comm,
        dealii::Triangulation<dim>::none,
        parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    }
    else if(data.triangulation_type == TriangulationType::FullyDistributed)
    {
      triangulation = std::make_shared<parallel::fullydistributed::Triangulation<dim>>(mpi_comm);
    }
    else
    {
      AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
    }

    // mapping
    mapping = std::make_shared<MappingQGeneric<dim>>(data.mapping_degree);
  }

  /**
   * Attach a pointer for moving grid functionality.
   */
  void
  attach_grid_motion(std::shared_ptr<GridMotionInterface<dim, Number>> grid_motion_in)
  {
    grid_motion = grid_motion_in;
  }

  /**
   * Return pointer to static mapping.
   */
  std::shared_ptr<Mapping<dim> const>
  get_static_mapping() const
  {
    AssertThrow(mapping.get() != 0, ExcMessage("Grid::mapping is uninitialized."));

    return mapping;
  }

  /**
   * Return pointer to dynamic mapping (and redirect to static mapping if dynamic mapping is not
   * initialized).
   */
  std::shared_ptr<Mapping<dim> const>
  get_dynamic_mapping() const
  {
    if(grid_motion.get() != 0)
      return grid_motion->get_mapping();
    else
      return mapping;
  }

  /**
   * dealii::Triangulation.
   */
  std::shared_ptr<Triangulation<dim>> triangulation;

  /**
   * dealii::GridTools::PeriodicFacePair's.
   */
  PeriodicFaces periodic_faces;

  /**
   * dealii::Mapping. Describes reference configuration.
   */
  std::shared_ptr<Mapping<dim>> mapping;

  /**
   * Computes and describes dynamic grid motion.
   */
  std::shared_ptr<GridMotionInterface<dim, Number>> grid_motion;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
