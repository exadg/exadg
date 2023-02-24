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

#ifndef INCLUDE_EXADG_GRID_GRID_UTILITIES_H_
#define INCLUDE_EXADG_GRID_GRID_UTILITIES_H_

// deal.II
#include <deal.II/grid/grid_in.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// ExaDG
#include <exadg/grid/balanced_granularity_partition_policy.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/perform_local_refinements.h>

namespace ExaDG
{
namespace GridUtilities
{
/**
 * Returns the type of elements, where we currently only allow triangulations consisting of the same
 * type of elements.
 */
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
 * Returns the type of dealii mesh smoothing depending on the element type and whether we use
 * local-smoothing multigrid.
 */
template<int dim>
typename dealii::Triangulation<dim>::MeshSmoothing
get_mesh_smoothing(bool const use_local_smoothing_multigrid, ElementType const & element_type)
{
  typename dealii::Triangulation<dim>::MeshSmoothing mesh_smoothing;

  if(element_type == ElementType::Simplex)
  {
    mesh_smoothing = dealii::Triangulation<dim>::none;

    // the option limit_level_difference_at_vertices (required for local smoothing multigrid) is not
    // implemented for simplicial elements.
  }
  else if(element_type == ElementType::Hypercube)
  {
    if(use_local_smoothing_multigrid)
      mesh_smoothing = dealii::Triangulation<dim>::limit_level_difference_at_vertices;
    else
      mesh_smoothing = dealii::Triangulation<dim>::none;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return mesh_smoothing;
}

/**
 * This function creates a triangulation based on a lambda function and refinement parameters for
 * global and local mesh refinements. This function is used to create the fine triangulation on the
 * one hand and the coarse triangulations required for global-coarsening multigrid on the other
 * hand.
 *
 * This function expects that the argument tria has already been constructed.
 */
template<int dim>
inline void
create_triangulation(
  std::shared_ptr<dealii::Triangulation<dim>>                    tria,
  GridData const &                                               data,
  std::function<void(dealii::Triangulation<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  unsigned int const                                             global_refinements,
  std::vector<unsigned int> const & vector_local_refinements = std::vector<unsigned int>())
{
  if(data.element_type == ElementType::Simplex)
  {
    AssertThrow(
      vector_local_refinements.size() == 0,
      dealii::ExcMessage(
        "Currently, dealii triangulations composed of simplicial elements do not allow local refinements."));
  }

  if(data.triangulation_type == TriangulationType::Serial or
     data.triangulation_type == TriangulationType::Distributed)
  {
    lambda_create_triangulation(*tria, global_refinements, vector_local_refinements);
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    auto const serial_grid_generator = [&](dealii::Triangulation<dim, dim> & tria_serial) {
      lambda_create_triangulation(tria_serial, global_refinements, vector_local_refinements);
    };

    auto const serial_grid_partitioner = [&](dealii::Triangulation<dim, dim> & tria_serial,
                                             MPI_Comm const                    comm,
                                             unsigned int const                group_size) {
      (void)group_size;
      if(data.partitioning_type == PartitioningType::Metis)
      {
        dealii::GridTools::partition_triangulation(dealii::Utilities::MPI::n_mpi_processes(comm),
                                                   tria_serial);
      }
      else if(data.partitioning_type == PartitioningType::z_order)
      {
        dealii::GridTools::partition_triangulation_zorder(
          dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    };

    unsigned int const group_size = 1;

    typename dealii::TriangulationDescription::Settings triangulation_description_setting =
      dealii::TriangulationDescription::default_setting;

    if(data.element_type == ElementType::Simplex)
    {
      triangulation_description_setting = dealii::TriangulationDescription::default_setting;

      // the option construct_multigrid_hierarchy (required for local smoothing multigrid) is not
      // implemented for simplicial elements.
    }
    else if(data.element_type == ElementType::Hypercube)
    {
      if(data.multigrid == MultigridVariant::LocalSmoothing)
        triangulation_description_setting =
          dealii::TriangulationDescription::construct_multigrid_hierarchy;
      else
        triangulation_description_setting = dealii::TriangulationDescription::default_setting;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    auto const mesh_smoothing =
      GridUtilities::get_mesh_smoothing<dim>(data.multigrid == MultigridVariant::LocalSmoothing,
                                             data.element_type);

    auto const description = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(serial_grid_generator,
                                                                serial_grid_partitioner,
                                                                tria->get_communicator(),
                                                                group_size,
                                                                mesh_smoothing,
                                                                triangulation_description_setting);

    tria->create_triangulation(description);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
  }
}

/**
 * Given a fine_triangulation, this function creates all the coarse triangulations required for
 * global-coarsening multigrid
 */
template<int dim>
inline void
create_coarse_triangulations(
  dealii::Triangulation<dim> const &                               fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> & coarse_triangulations,
  GridData const &                                                 data,
  std::function<void(dealii::Triangulation<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const &   lambda_create_triangulation,
  std::vector<unsigned int> const                                  vector_local_refinements)
{
  // in case of a serial or distributed triangulation, deal.II can automatically generate the
  // coarse grid triangulations
  if(data.triangulation_type == TriangulationType::Serial)
  {
    AssertThrow(
      fine_triangulation.all_reference_cells_are_hyper_cube(),
      dealii::ExcMessage(
        "The create_geometric_coarsening_sequence function of dealii does currently not support "
        "simplicial elements."));

    coarse_triangulations =
      dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        fine_triangulation);
  }
  else if(data.triangulation_type == TriangulationType::Distributed)
  {
    AssertThrow(
      fine_triangulation.all_reference_cells_are_hyper_cube(),
      dealii::ExcMessage(
        "dealii::parallel::distributed::Triangulation does not support simplicial elements."));

    coarse_triangulations =
      dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        fine_triangulation,
        BalancedGranularityPartitionPolicy<dim>(
          dealii::Utilities::MPI::n_mpi_processes(fine_triangulation.get_communicator())));
  }
  else if(data.triangulation_type == TriangulationType::FullyDistributed)
  {
    // resize the empty coarse triangulations vector
    coarse_triangulations = std::vector<std::shared_ptr<dealii::Triangulation<dim> const>>(
      fine_triangulation.n_global_levels());

    // lambda function for creating the coarse triangulations
    auto const lambda_create_level_triangulation = [&](unsigned int              refine_global,
                                                       std::vector<unsigned int> refine_local) {
      auto level_tria = std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(
        fine_triangulation.get_communicator());

      GridUtilities::create_triangulation<dim>(
        level_tria, data, lambda_create_triangulation, refine_global, refine_local);

      return level_tria;
    };

    // we start with the finest level
    unsigned int              level        = fine_triangulation.n_global_levels() - 1;
    std::vector<unsigned int> refine_local = vector_local_refinements;

    // make the last entry of the coarse_triangulations point to the fine_triangulation
    coarse_triangulations[level].reset(&fine_triangulation, [](auto *) {
      // empty deleter, since fine_triangulation is an external field
      // and its destructor is called somewhere else
    });

    level--;

    // undo global refinements
    for(int refine_global = data.n_refine_global - 1; refine_global >= 0; --refine_global)
    {
      coarse_triangulations[level] = lambda_create_level_triangulation(refine_global, refine_local);

      level--;
    }

    // undo local refinements
    if(refine_local.size() > 0)
    {
      while(*std::max_element(refine_local.begin(), refine_local.end()) != 0)
      {
        for(size_t material_id = 0; material_id < refine_local.size(); material_id++)
        {
          if(refine_local[material_id] > 0)
          {
            refine_local[material_id]--;
          }
        }
        coarse_triangulations[level] = lambda_create_level_triangulation(0, refine_local);

        level--;
      }
    }
  }
}

/**
 * This function creates both the fine triangulation and, if needed, the coarse triangulations
 * required for global coarsening multigrid. In addition to the functions above, this function
 * exists in order to provide a simple interface for applications.
 */
template<int dim>
inline void
create_fine_and_coarse_triangulations(
  Grid<dim> &                                                    grid,
  GridData const &                                               data,
  bool const                                                     involves_h_multigrid,
  std::function<void(dealii::Triangulation<dim> &,
                     unsigned int const,
                     std::vector<unsigned int> const &)> const & lambda_create_triangulation,
  std::vector<unsigned int> const vector_local_refinements = std::vector<unsigned int>())
{
  GridUtilities::create_triangulation(grid.triangulation,
                                      data,
                                      lambda_create_triangulation,
                                      data.n_refine_global,
                                      vector_local_refinements);

  // coarse triangulations need to be created for global coarsening multigrid
  if(data.multigrid == MultigridVariant::GlobalCoarsening and involves_h_multigrid)
  {
    GridUtilities::create_coarse_triangulations(*grid.triangulation,
                                                grid.coarse_triangulations,
                                                data,
                                                lambda_create_triangulation,
                                                vector_local_refinements);
  }
}

/**
 * This function reads an external triangulation. The function takes GridData as an argument
 * and stores the external triangulation in "tria".
 */
template<int dim>
inline void
read_external_triangulation(dealii::Triangulation<dim, dim> & tria, GridData const & data)
{
  AssertThrow(!data.file_name.empty(),
              dealii::ExcMessage(
                "You are trying to read a grid file, but the string, which is supposed to contain"
                " the file name, is empty."));

  dealii::GridIn<dim> grid_in;

  grid_in.attach_triangulation(tria);

  // find the file extension from the file name
  std::string extension = data.file_name.substr(data.file_name.find_last_of('.') + 1);

  AssertThrow(!extension.empty(),
              dealii::ExcMessage("You are trying to read a grid file, but the file extension is"
                                 " empty."));

  // decide the file format
  typename dealii::GridIn<dim>::Format format;
  if(extension == "e" || extension == "exo")
    format = dealii::GridIn<dim>::Format::exodusii;
  else
    format = grid_in.parse_format(extension);

  // TODO: check if the exodusIIData is needed
  // typename dealii::GridIn<dim>::ExodusIIData exodusIIData;

  grid_in.read(data.file_name, format);

  AssertThrow(get_element_type(tria) == data.element_type,
              dealii::ExcMessage("You are trying to read a grid file, but the element type of the"
                                 " external grid file and the element type specified in GridData"
                                 " don't match. Most likely, you forgot to change the element_type"
                                 " parameter of GridData to the desired element type."));
}

} // namespace GridUtilities
} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_UTILITIES_H_ */
