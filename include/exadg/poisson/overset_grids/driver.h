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

#ifndef INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_
#define INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/poisson/overset_grids/user_interface/application_base.h>
#include <exadg/poisson/spatial_discretization/operator.h>

namespace ExaDG
{
namespace Poisson
{
namespace OversetGrids
{
template<int dim, int n_components, typename Number>
class Solver
{
public:
  void
  setup(std::shared_ptr<Domain<dim, n_components, Number> const> const & domain,
        std::shared_ptr<Grid<dim> const> const &                         grid,
        std::shared_ptr<dealii::Mapping<dim> const> const &              mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> const            multigrid_mappings,
        MPI_Comm const                                                   mpi_comm)
  {
    pde_operator =
      std::make_shared<Operator<dim, n_components, Number>>(grid,
                                                            mapping,
                                                            multigrid_mappings,
                                                            domain->get_boundary_descriptor(),
                                                            domain->get_field_functions(),
                                                            domain->get_parameters(),
                                                            "Poisson",
                                                            mpi_comm);

    pde_operator->setup();

    postprocessor = domain->create_postprocessor();
    postprocessor->setup(*pde_operator);
  }

  std::shared_ptr<Operator<dim, n_components, Number>>          pde_operator;
  std::shared_ptr<PostProcessorBase<dim, n_components, Number>> postprocessor;
};

template<int dim, int n_components, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const &                                            mpi_comm,
         std::shared_ptr<ApplicationBase<dim, n_components, Number>> application);

  void
  setup();

  void
  solve();

private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  std::shared_ptr<ApplicationBase<dim, n_components, Number>> application;

  std::shared_ptr<Grid<dim>> grid1, grid2;

  std::shared_ptr<dealii::Mapping<dim>> mapping1, mapping2;

  std::shared_ptr<MultigridMappings<dim, Number>> multigrid_mappings1, multigrid_mappings2;

  // Poisson solvers
  std::shared_ptr<Solver<dim, n_components, Number>> poisson1, poisson2;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<rank, dim, Number>> first_to_second, second_to_first;
};
} // namespace OversetGrids
} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_ */
