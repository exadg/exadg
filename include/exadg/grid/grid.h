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
#include <exadg/grid/grid_data.h>

namespace ExaDG
{
using namespace dealii;

template<int dim>
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
   * dealii::Triangulation.
   */
  std::shared_ptr<Triangulation<dim>> triangulation;

  /**
   * dealii::GridTools::PeriodicFacePair's.
   */
  PeriodicFaces periodic_faces;

  /**
   * dealii::Mapping.
   */
  std::shared_ptr<Mapping<dim>> mapping;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
