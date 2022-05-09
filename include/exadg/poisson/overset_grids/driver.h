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
#include <exadg/poisson/solver_poisson.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class DriverOversetGrids
{
public:
  DriverOversetGrids(
    MPI_Comm const &                                                        mpi_comm,
    std::shared_ptr<ApplicationOversetGridsBase<dim, n_components, Number>> application);

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

  std::shared_ptr<ApplicationOversetGridsBase<dim, n_components, Number>> application;

  // Poisson solvers
  std::shared_ptr<SolverPoisson<dim, n_components, Number>> poisson1, poisson2;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<rank, dim, Number>> first_to_second, second_to_first;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_ */
