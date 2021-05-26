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

#include <exadg/grid/enum_types.h>

namespace ExaDG
{
struct GridData
{
  GridData()
    : triangulation_type(TriangulationType::Distributed), n_refine_global(0), mapping_degree(1)
  {
  }

  TriangulationType triangulation_type;

  unsigned int n_refine_global;

  unsigned int mapping_degree;

  // TODO: path to a grid file
  // std::string grid_file;
};

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
    if(data.triangulation_type == TriangulationType::Distributed)
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
    mapping.reset(new MappingQGeneric<dim>(data.mapping_degree));
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
