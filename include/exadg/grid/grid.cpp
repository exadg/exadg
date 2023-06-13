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

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/perform_local_refinements.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
template<int dim>
void
Grid<dim>::initialize(GridData const & data, MPI_Comm const & mpi_comm)
{
  // triangulation
  if(data.triangulation_type == TriangulationType::Serial)
  {
    auto mesh_smoothing = dealii::Triangulation<dim>::none;

    if(not data.create_coarse_triangulations)
      mesh_smoothing = dealii::Triangulation<dim>::limit_level_difference_at_vertices;

    AssertDimension(dealii::Utilities::MPI::n_mpi_processes(mpi_comm), 1);
    triangulation = std::make_shared<dealii::Triangulation<dim>>(mesh_smoothing);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    auto mesh_smoothing = dealii::Triangulation<dim>::none;
    typename dealii::parallel::distributed::Triangulation<dim>::Settings distributed_settings;

    // TODO It seems as if we set these values in case where we do not want/need this.
    // Assume for example one wants to use only p-multigrid or no multigrid at all. In that case,
    // it seems as if construct_multigrid_hierarchy is set unnecessarily.
    if(not data.create_coarse_triangulations)
    {
      mesh_smoothing = dealii::Triangulation<dim>::limit_level_difference_at_vertices;
      distributed_settings =
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy;
    }

    triangulation =
      std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_comm,
                                                                          mesh_smoothing,
                                                                          distributed_settings);
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation =
      std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_comm);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }
}

template class Grid<2>;
template class Grid<3>;

} // namespace ExaDG
