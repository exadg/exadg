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
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/grid/grid_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>

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
  BalancedGranularityPartitionPolicy(unsigned int const n_mpi_processes);

  virtual ~BalancedGranularityPartitionPolicy(){};

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  partition(dealii::Triangulation<dim, spacedim> const & tria_coarse_in) const override;

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
  Grid(GridData const & data, MPI_Comm const & mpi_comm);

  void
  create_triangulation(
    GridData const &                                          data,
    std::function<void(dealii::Triangulation<dim> &)> const & create_coarse_triangulation,
    unsigned int const                                        global_refinements,
    std::vector<unsigned int> const & vector_local_refinements = std::vector<unsigned int>());

  std::shared_ptr<dealii::Triangulation<dim> const>
  get_triangulation() const;

  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const &
  get_coarse_triangulations() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

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
    std::vector<unsigned int> const &                         vector_local_refinements);
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_GRID_H_ */
