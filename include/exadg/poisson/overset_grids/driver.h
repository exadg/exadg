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
template<int dim, int n_components, typename Number>
class Solver
{
public:
  void
  setup(std::shared_ptr<Domain<dim, n_components, Number>> domain, MPI_Comm const mpi_comm)
  {
    pde_operator =
      std::make_shared<Operator<dim, n_components, Number>>(domain->get_grid(),
                                                            domain->get_mapping(),
                                                            domain->get_boundary_descriptor(),
                                                            domain->get_field_functions(),
                                                            domain->get_parameters(),
                                                            "Poisson",
                                                            mpi_comm);

    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
    matrix_free_data->append(pde_operator);

    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    if(domain->get_parameters().enable_cell_based_face_loops)
      Categorization::do_cell_based_loops(*domain->get_grid()->triangulation,
                                          matrix_free_data->data);
    matrix_free->reinit(*domain->get_mapping(),
                        matrix_free_data->get_dof_handler_vector(),
                        matrix_free_data->get_constraint_vector(),
                        matrix_free_data->get_quadrature_vector(),
                        matrix_free_data->data);

    pde_operator->setup(matrix_free, matrix_free_data);

    pde_operator->setup_solver();

    postprocessor = domain->create_postprocessor();
    postprocessor->setup(*pde_operator);
  }

  std::shared_ptr<Operator<dim, n_components, Number>>          pde_operator;
  std::shared_ptr<PostProcessorBase<dim, n_components, Number>> postprocessor;

private:
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
};

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
  std::shared_ptr<Solver<dim, n_components, Number>> poisson1, poisson2;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<rank, dim, Number>> first_to_second, second_to_first;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_DRIVER_H_ */
