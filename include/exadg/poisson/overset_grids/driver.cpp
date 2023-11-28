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

// ExaDG
#include <exadg/poisson/overset_grids/driver.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Poisson
{
namespace OversetGrids
{
template<int dim, int n_components, typename Number>
Driver<dim, n_components, Number>::Driver(
  MPI_Comm const &                                            comm,
  std::shared_ptr<ApplicationBase<dim, n_components, Number>> app)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, false /* is_test */);

  poisson1 = std::make_shared<Solver<dim, n_components, Number>>();
  poisson2 = std::make_shared<Solver<dim, n_components, Number>>();
}

template<int dim, int n_components, typename Number>
void
Driver<dim, n_components, Number>::setup()
{
  pcout << std::endl << "Setting up Poisson solver for overset grids:" << std::endl;

  // setup application
  AssertThrow(application->domain1.get(), dealii::ExcMessage("Domain 1 is uninitialized."));
  AssertThrow(application->domain2.get(), dealii::ExcMessage("Domain 2 is uninitialized."));

  application->domain1->setup_pre(grid1, mapping1, {"Domain1"});
  application->domain2->setup_pre(grid2, mapping2, {"Domain2"});

  // set boundary IDs for domain 1
  set_boundary_ids_overlap_region(*grid1->triangulation,
                                  application->boundary_id_overlap,
                                  *mapping2,
                                  *grid2->triangulation);

  // set boundary IDs for domain 2
  set_boundary_ids_overlap_region(*grid2->triangulation,
                                  application->boundary_id_overlap,
                                  *mapping1,
                                  *grid1->triangulation);

  application->domain1->setup_post(grid1);
  application->domain2->setup_post(grid2);

  // setup Poisson solvers
  poisson1->setup(application->domain1, grid1, mapping1, mpi_comm);
  poisson2->setup(application->domain2, grid2, mapping2, mpi_comm);

  // setup interface coupling
  {
    // domain 1 to domain 2
    pcout << std::endl << "Setup interface coupling first -> second ..." << std::endl;

    first_to_second = std::make_shared<InterfaceCoupling<rank, dim, Number>>();
    // No map of boundary IDs can be provided to make the search more efficient. The reason behind
    // is that the two domains are not connected along boundaries but are overlapping instead. To
    // resolve this, the implementation of InterfaceCoupling needs to be generalized.
    first_to_second->setup(poisson2->pde_operator->get_container_interface_data(),
                           poisson1->pde_operator->get_dof_handler(),
                           *mapping1,
                           {} /* marked_vertices */,
                           1.e-8 /* geometric tolerance */);

    pcout << std::endl << "... done." << std::endl;

    // domain 2 to domain 1
    pcout << std::endl << "Setup interface coupling second -> first ..." << std::endl;

    second_to_first = std::make_shared<InterfaceCoupling<rank, dim, Number>>();
    second_to_first->setup(poisson1->pde_operator->get_container_interface_data(),
                           poisson2->pde_operator->get_dof_handler(),
                           *mapping2,
                           {} /* marked_vertices */,
                           1.e-8 /* geometric tolerance */);

    pcout << std::endl << "... done." << std::endl;
  }
}

template<int dim, int n_components, typename Number>
void
Driver<dim, n_components, Number>::solve()
{
  // initialization of vectors
  dealii::LinearAlgebra::distributed::Vector<Number> rhs_1, rhs_2;
  dealii::LinearAlgebra::distributed::Vector<Number> sol_1, sol_2;
  poisson1->pde_operator->initialize_dof_vector(rhs_1);
  poisson1->pde_operator->initialize_dof_vector(sol_1);
  poisson1->pde_operator->prescribe_initial_conditions(sol_1);

  poisson2->pde_operator->initialize_dof_vector(rhs_2);
  poisson2->pde_operator->initialize_dof_vector(sol_2);
  poisson2->pde_operator->prescribe_initial_conditions(sol_2);

  // postprocessing of results
  poisson1->postprocessor->do_postprocessing(sol_1);
  poisson2->postprocessor->do_postprocessing(sol_2);

  // solve linear system of equations
  bool         converged = false;
  unsigned int iter      = 0;
  while(not(converged))
  {
    // solve on domain 1
    poisson1->pde_operator->rhs(rhs_1);
    poisson1->pde_operator->solve(sol_1, rhs_1, 0.0 /* time */);

    // Transfer data domain 1 to domain 2
    first_to_second->update_data(sol_1);

    // solve on domain 2
    poisson2->pde_operator->rhs(rhs_2);
    poisson2->pde_operator->solve(sol_2, rhs_2, 0.0 /* time */);

    // Transfer data from 2 to 1
    second_to_first->update_data(sol_2);

    // postprocessing of results
    poisson1->postprocessor->do_postprocessing(sol_1);
    poisson2->postprocessor->do_postprocessing(sol_2);

    // TODO check convergence
    ++iter;
    if(iter > 10)
      converged = true;
  }
}

template class Driver<2, 1, float>;
template class Driver<3, 1, float>;
template class Driver<2, 2, float>;
template class Driver<3, 3, float>;

template class Driver<2, 1, double>;
template class Driver<3, 1, double>;
template class Driver<2, 2, double>;
template class Driver<3, 3, double>;

} // namespace OversetGrids
} // namespace Poisson
} // namespace ExaDG
