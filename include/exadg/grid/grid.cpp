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

// deal.II
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/perform_local_refinements.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
template<int dim>
Grid<dim>::Grid(const GridData & data, MPI_Comm const & mpi_comm)
{
  if(data.element_type == ElementType::Simplex)
  {
    mesh_smoothing = dealii::Triangulation<dim>::none;

    // the option limit_level_difference_at_vertices (required for local smoothing multigrid) is not
    // implemented for simplicial elements.
  }
  else if(data.element_type == ElementType::Hypercube)
  {
    if(data.create_coarse_triangulations) // global coarsening multigrid
      mesh_smoothing = dealii::Triangulation<dim>::none;
    else // required for local smoothing
      mesh_smoothing = dealii::Triangulation<dim>::limit_level_difference_at_vertices;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  // triangulation
  if(data.triangulation_type == TriangulationType::Serial)
  {
    AssertDimension(dealii::Utilities::MPI::n_mpi_processes(mpi_comm), 1);
    triangulation = std::make_shared<dealii::Triangulation<dim>>(mesh_smoothing);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    typename dealii::parallel::distributed::Triangulation<dim>::Settings distributed_settings;

    if(data.create_coarse_triangulations)
      distributed_settings = dealii::parallel::distributed::Triangulation<dim>::default_setting;
    else
      distributed_settings =
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy;

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

  // mapping
  if(data.element_type == ElementType::Hypercube)
  {
    mapping = std::make_shared<dealii::MappingQ<dim>>(data.mapping_degree);
  }
  else if(data.element_type == ElementType::Simplex)
  {
    mapping =
      std::make_shared<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(data.mapping_degree));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter element_type."));
  }
}

template class Grid<2>;
template class Grid<3>;

} // namespace ExaDG
