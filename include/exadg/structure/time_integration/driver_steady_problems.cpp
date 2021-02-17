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
#include <exadg/structure/user_interface/input_parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
DriverSteady<dim, Number>::DriverSteady(std::shared_ptr<Interface::Operator<Number>> operator_in,
                                        std::shared_ptr<PostProcessorBase<Number>> postprocessor_in,
                                        InputParameters const &                    param_in,
                                        MPI_Comm const &                           mpi_comm_in,
                                        bool const print_wall_times_in)
  : pde_operator(operator_in),
    postprocessor(postprocessor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    print_wall_times(print_wall_times_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
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
DriverSteady<dim, Number>::solve_problem()
{
  Timer timer;
  timer.restart();

  postprocessing();

  solve();

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
DriverSteady<dim, Number>::solve()
{
  Timer timer;
  timer.restart();

  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  if(param.large_deformation) // nonlinear problem
  {
    VectorType const_vector;
    auto const iter = pde_operator->solve_nonlinear(
      solution, const_vector, 0.0 /* no mass term */, 0.0 /* time */, param.update_preconditioner);

    unsigned int const N_iter_nonlinear = std::get<0>(iter);
    unsigned int const N_iter_linear    = std::get<1>(iter);

    print_solver_info_nonlinear(
      pcout, N_iter_nonlinear, N_iter_linear, timer.wall_time(), print_wall_times);
  }
  else // linear problem
  {
    // calculate right-hand side vector
    pde_operator->compute_rhs_linear(rhs_vector, 0.0 /* time */);

    unsigned int const N_iter_linear =
      pde_operator->solve_linear(solution, rhs_vector, 0.0 /* no mass term */, 0.0 /* time */);

    print_solver_info_linear(pcout, N_iter_linear, timer.wall_time(), print_wall_times);
  }

  pcout << std::endl << "... done!" << std::endl;

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::postprocessing() const
{
  Timer timer;
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
