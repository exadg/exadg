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

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/calculate_maximum_aspect_ratio.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/application_base.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/solver_result.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number>
class DriverOversetGrids
{
public:
  DriverOversetGrids(MPI_Comm const &                                          mpi_comm,
                     std::shared_ptr<ApplicationOversetGridsBase<dim, Number>> application);

  void
  setup();

  void
  solve();

private:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationOversetGridsBase<dim, Number>> application;

  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free, matrix_free_second;
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data, matrix_free_data_second;

  std::shared_ptr<Operator<dim, Number, dim>> pde_operator, pde_operator_second;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor, postprocessor_second;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> first_to_second, second_to_first;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_ */
