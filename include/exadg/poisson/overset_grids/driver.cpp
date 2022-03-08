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
#include <exadg/utilities/throughput_parameters.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number>
DriverOversetGrids<dim, Number>::DriverOversetGrids(
  MPI_Comm const &                                               comm,
  std::shared_ptr<ApplicationOversetGridsBase<dim, dim, Number>> app)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, false /* is_test */);

  poisson1 = std::make_shared<SolverPoisson<dim, Number>>();
  poisson2 = std::make_shared<SolverPoisson<dim, Number>>();
}

template<int dim, typename Number>
void
DriverOversetGrids<dim, Number>::setup()
{
  pcout << std::endl << "Setting up Poisson solver for overset grids:" << std::endl;

  application->setup();

  // setup Poisson solvers
  poisson1->setup(application->domain1, mpi_comm, false /* is_test */);
  poisson2->setup(application->domain2, mpi_comm, false /* is_test */);

  // setup interface coupling
  {
    // main domain to domain 2
    pcout << std::endl << "Setup interface coupling first -> second ..." << std::endl;

    first_to_second = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    // No map of boundary IDs can be provided to make the search more efficient. The reason behind
    // is that the two domains are not connected along boundaries but are overlapping instead. To
    // resolve this, the implementation of InterfaceCoupling needs to be generalized.
    std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim, double>>> dummy;
    first_to_second->setup(poisson2->pde_operator->get_container_interface_data(),
                           dummy,
                           poisson1->pde_operator->get_dof_handler(),
                           *application->domain1->get_grid()->mapping,
                           1.e-8 /* geometric tolerance */);


    // domain 2 to domain 1
    pcout << std::endl << "Setup interface coupling second -> first ..." << std::endl;

    second_to_first = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    // TODO
    //    second_to_first->setup(
    //      poisson1->pde_operator->get_container_interface_data(),
    //      dummy,
    //      poisson2->pde_operator->get_dof_handler(),
    //      *application->domain2->get_grid()->mapping,
    //      1.e-8 /* geometric tolerance */);
  }
}

template<int dim, typename Number>
void
DriverOversetGrids<dim, Number>::solve()
{
  // initialization of vectors
  dealii::LinearAlgebra::distributed::Vector<Number> rhs, rhs_2;
  dealii::LinearAlgebra::distributed::Vector<Number> sol, sol_2;
  poisson1->pde_operator->initialize_dof_vector(rhs);
  poisson1->pde_operator->initialize_dof_vector(sol);
  poisson1->pde_operator->prescribe_initial_conditions(sol);

  poisson2->pde_operator->initialize_dof_vector(rhs_2);
  poisson2->pde_operator->initialize_dof_vector(sol_2);
  poisson2->pde_operator->prescribe_initial_conditions(sol_2);

  // postprocessing of results
  poisson1->postprocessor->do_postprocessing(sol);
  poisson2->postprocessor->do_postprocessing(sol_2);

  // solve linear system of equations
  bool         converged = false;
  unsigned int iter      = 0;
  while(not(converged))
  {
    // calculate right-hand side
    poisson1->pde_operator->rhs(rhs);
    poisson1->pde_operator->solve(sol, rhs, 0.0 /* time */);

    // Transfer data domain 1 to domain 2
    first_to_second->update_data(sol);

    // calculate right-hand side
    poisson1->pde_operator->rhs(rhs_2);
    poisson1->pde_operator->solve(sol_2, rhs_2, 0.0 /* time */);

    // Transfer data from 2 to 1
    // TODO
    //    second_to_first->update_data(sol);

    // postprocessing of results
    // TODO this could be shifted outside the loop
    // currently located within the loop to check convergence visually
    poisson1->postprocessor->do_postprocessing(sol);
    poisson1->postprocessor->do_postprocessing(sol_2);

    // TODO check convergence
    ++iter;
    if(iter > 10)
      converged = true;
  }
}

template class DriverOversetGrids<2, float>;
template class DriverOversetGrids<3, float>;

template class DriverOversetGrids<2, double>;
template class DriverOversetGrids<3, double>;

} // namespace Poisson
} // namespace ExaDG
