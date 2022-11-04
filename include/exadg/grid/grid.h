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
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/grid/perform_local_refinements.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
/**
 * A class to use for the deal.II coarsening functionality, where we try to
 * balance the mesh coarsening with a minimum granularity and the number of
 * partitions on coarser levels.
 */
template<int dim, int spacedim = dim>
class BalancedGranularityPartitionPolicy
  : public dealii::RepartitioningPolicyTools::Base<dim, spacedim>
{
public:
  BalancedGranularityPartitionPolicy(unsigned int const n_mpi_processes)
    : n_mpi_processes_per_level{n_mpi_processes}
  {
  }

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  partition(dealii::Triangulation<dim, spacedim> const & tria_coarse_in) const override
  {
    dealii::types::global_cell_index const n_cells = tria_coarse_in.n_global_active_cells();

    // TODO: We hard-code a grain-size limit of 200 cells per processor
    // (assuming linear finite elements and typical behavior of
    // supercomputers). In case we have fewer cells on the fine level, we do
    // not immediately go to 200 cells per rank, but limit the growth by a
    // factor of 8, which limits makes sure that we do not create too many
    // messages for individual MPI processes.
    unsigned int const grain_size_limit =
      std::min<unsigned int>(200, 8 * n_cells / n_mpi_processes_per_level.back() + 1);

    dealii::RepartitioningPolicyTools::MinimalGranularityPolicy<dim, spacedim> partitioning_policy(
      grain_size_limit);
    dealii::LinearAlgebra::distributed::Vector<double> const partitions =
      partitioning_policy.partition(tria_coarse_in);

    // The vector 'partitions' contains the partition numbers. To get the
    // number of partitions, we take the infinity norm.
    n_mpi_processes_per_level.push_back(static_cast<unsigned int>(partitions.linfty_norm()) + 1);
    return partitions;
  }

private:
  mutable std::vector<unsigned int> n_mpi_processes_per_level;
};

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

  void
  create_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_coarse_triangulation,
    unsigned int const                                        global_refinements,
    std::vector<unsigned int> const & vector_local_refinements = std::vector<unsigned int>())
  {
    // coarse triangulations need to be created for global coarsening multigrid
    if(data.create_coarse_triangulations)
    {
      if(data.triangulation_type == TriangulationType::Serial or
         data.triangulation_type == TriangulationType::Distributed)
      {
        // in case of a serial or distributed triangulation, deal.II can automatically generate the
        // coarse grid triangulations

        do_create_triangulation(data,
                                create_coarse_triangulation,
                                global_refinements,
                                vector_local_refinements);

        coarse_triangulations =
          dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
            *triangulation,
            BalancedGranularityPartitionPolicy<dim>(
              dealii::Utilities::MPI::n_mpi_processes(triangulation->get_communicator())));
      }
      else if(data.triangulation_type == TriangulationType::FullyDistributed)
      {
        // we need to generate the coarse triangulations explicitly

        // TODO
        // construct all the coarser triangulations

        AssertThrow(false, ExcNotImplemented());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Invalid parameter triangulation_type."));
      }
    }
    else
    {
      do_create_triangulation(data,
                              create_coarse_triangulation,
                              global_refinements,
                              vector_local_refinements);
    }
  }

  std::shared_ptr<dealii::Triangulation<dim> const>
  get_triangulation() const
  {
    return triangulation;
  }

  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const &
  get_coarse_triangulations() const
  {
    return coarse_triangulations;
  }

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    return mapping;
  }

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

private:
  void
  do_create_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_triangulation,
    unsigned int const                                        global_refinements,
    std::vector<unsigned int> const &                         vector_local_refinements)
  {
    if(data.triangulation_type == TriangulationType::Serial or
       data.triangulation_type == TriangulationType::Distributed)
    {
      create_triangulation(*triangulation);

      if(vector_local_refinements.size() > 0)
        refine_local(*triangulation, vector_local_refinements);

      if(global_refinements > 0)
        triangulation->refine_global(global_refinements);
    }
    else if(data.triangulation_type == TriangulationType::FullyDistributed)
    {
      auto const serial_grid_generator = [&](dealii::Triangulation<dim, dim> & tria_serial) {
        create_triangulation(tria_serial);

        if(vector_local_refinements.size() > 0)
          refine_local(tria_serial, vector_local_refinements);

        if(global_refinements > 0)
          tria_serial.refine_global(global_refinements);
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
