/*
 * balanced_granularity_partition_policy.h
 *
 *  Created on: Jan 26, 2023
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_GRID_BALANCED_GRANULARITY_PARTITION_POLICY_H_
#define INCLUDE_EXADG_GRID_BALANCED_GRANULARITY_PARTITION_POLICY_H_

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

  virtual ~BalancedGranularityPartitionPolicy(){};

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
} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_BALANCED_GRANULARITY_PARTITION_POLICY_H_ */
