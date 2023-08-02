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

#include <exadg/structure/postprocessor/postprocessor_base.h>
#include <exadg/structure/spatial_discretization/interface.h>
#include <exadg/structure/time_integration/driver_steady_problems.h>
#include <exadg/structure/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
DriverSteady<dim, Number>::DriverSteady(std::shared_ptr<Interface::Operator<Number>> operator_,
                                        std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
                                        Parameters const &                           param_,
                                        MPI_Comm const &                             mpi_comm_,
                                        bool const                                   is_test_)
  : pde_operator(operator_),
    postprocessor(postprocessor_),
    param(param_),
    mpi_comm(mpi_comm_),
    is_test(is_test_),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    timer_tree(new TimerTree())
{
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::solve()
{
  dealii::Timer timer;
  timer.restart();

  postprocessing();

  do_solve();

  postprocessing();

  timer_tree->insert({"DriverSteady"}, timer.wall_time());
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
DriverSteady<dim, Number>::get_timings() const
{
  return timer_tree;
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(solution);
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_displacement(solution, time);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::do_solve()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  if(param.large_deformation) // nonlinear problem
  {
    VectorType const const_vector_dummy; // will not be used
    auto const       iter = pde_operator->solve_nonlinear(solution,
                                                    const_vector_dummy,
                                                    0.0 /* no acceleration term */,
                                                    0.0 /* no damping term */,
                                                    0.0 /* time */,
                                                    param.update_preconditioner);

    unsigned int const N_iter_nonlinear = std::get<0>(iter);
    unsigned int const N_iter_linear    = std::get<1>(iter);

    if(not(is_test))
      print_solver_info_nonlinear(pcout, N_iter_nonlinear, N_iter_linear, timer.wall_time());
  }
  else // linear problem
  {
    // calculate right-hand side vector
    pde_operator->rhs(rhs_vector, 0.0 /* time */);

    unsigned int const N_iter_linear =
      pde_operator->solve_linear(solution,
                                 rhs_vector,
                                 0.0 /* no acceleration term */,
                                 0.0 /* no damping term */,
                                 0.0 /* time */,
                                 false /* update preconditioner */);

    if(not(is_test))
      print_solver_info_linear(pcout, N_iter_linear, timer.wall_time());
  }

  pcout << std::endl << "... done!" << std::endl;

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(solution);

  timer_tree->insert({"DriverSteady", "Postprocessing"}, timer.wall_time());
}

template class DriverSteady<2, float>;
template class DriverSteady<2, double>;

template class DriverSteady<3, float>;
template class DriverSteady<3, double>;

} // namespace Structure
} // namespace ExaDG
