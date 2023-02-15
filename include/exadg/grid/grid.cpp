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
Grid<dim>::Grid(GridData const & data, bool const involves_h_multigrid, MPI_Comm const & mpi_comm)
{
  auto const mesh_smoothing = GridUtilities::get_mesh_smoothing<dim>(
    data.multigrid == MultigridVariant::LocalSmoothing and involves_h_multigrid, data.element_type);

  // triangulation
  if(data.triangulation_type == TriangulationType::Serial)
  {
    AssertDimension(dealii::Utilities::MPI::n_mpi_processes(mpi_comm), 1);
    triangulation = std::make_shared<dealii::Triangulation<dim>>(mesh_smoothing);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    typename dealii::parallel::distributed::Triangulation<dim>::Settings distributed_settings;

    if(data.multigrid == MultigridVariant::LocalSmoothing and involves_h_multigrid)
      distributed_settings =
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy;
    else
      distributed_settings = dealii::parallel::distributed::Triangulation<dim>::default_setting;

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
