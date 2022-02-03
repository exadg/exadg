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
#include <exadg/grid/perform_local_refinements.h>

namespace ExaDG
{
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
  Grid(GridData const & data, MPI_Comm const & mpi_comm)
  {
    // triangulation
    if(data.triangulation_type == TriangulationType::Serial)
    {
      AssertDimension(dealii::Utilities::MPI::n_mpi_processes(mpi_comm), 1);
      triangulation = std::make_shared<dealii::Triangulation<dim>>();
    }
    else if(data.triangulation_type == TriangulationType::Distributed)
    {
      triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
        mpi_comm,
        dealii::Triangulation<dim>::none,
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
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
    // TODO SIMPLEX: this will not work in case of simplex meshes (-> use MappingFE)
    mapping = std::make_shared<dealii::MappingQGeneric<dim>>(data.mapping_degree);
  }

  void
  create_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_coarse_triangulation,
    std::vector<unsigned int> const & vector_local_refinements = std::vector<unsigned int>())
  {
    do_create_triangulation(data,
                            create_coarse_triangulation,
                            true /* do refine */,
                            vector_local_refinements);
  }

  void
  create_but_do_not_refine_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_fine_triangulation)
  {
    do_create_triangulation(data, create_fine_triangulation, false /* do not refine */);
  }

  /**
   * dealii::Triangulation.
   */
  std::shared_ptr<dealii::Triangulation<dim>> triangulation;

  /**
   * dealii::GridTools::PeriodicFacePair's.
   */
  PeriodicFaces periodic_faces;

  /**
   * dealii::Mapping.
   */
  std::shared_ptr<dealii::Mapping<dim>> mapping;

private:
  void
  do_create_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_triangulation,
    bool const                                                perform_refinements,
    std::vector<unsigned int> const &                         vector_local_refinements)
  {
    if(data.triangulation_type == TriangulationType::Serial or
       data.triangulation_type == TriangulationType::Distributed)
    {
      create_triangulation(*triangulation);

      if(perform_refinements)
      {
        if(vector_local_refinements.size() > 0)
          refine_local(*triangulation, vector_local_refinements);

        triangulation->refine_global(data.n_refine_global);
      }
    }
    else if(data.triangulation_type == TriangulationType::FullyDistributed)
    {
      auto const serial_grid_generator = [&](dealii::Triangulation<dim, dim> & tria_serial) {
        create_triangulation(tria_serial);

        if(perform_refinements)
        {
          if(vector_local_refinements.size() > 0)
            refine_local(tria_serial, vector_local_refinements);

          tria_serial.refine_global(data.n_refine_global);
        }
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

      // TODO SIMPLEX: this will not work in case of simplex meshes
      auto const description = dealii::TriangulationDescription::Utilities::
        create_description_from_triangulation_in_groups<dim, dim>(
          serial_grid_generator,
          serial_grid_partitioner,
          triangulation->get_communicator(),
          group_size,
          dealii::Triangulation<dim>::none,
          dealii::TriangulationDescription::construct_multigrid_hierarchy);

      triangulation->create_triangulation(description);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
    }
  }
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
