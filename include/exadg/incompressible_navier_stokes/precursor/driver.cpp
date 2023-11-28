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

#include <exadg/incompressible_navier_stokes/precursor/driver.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
namespace Precursor
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    application(app),
    use_adaptive_time_stepping(false)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  AssertThrow(application->precursor_is_active(),
              dealii::ExcMessage("One should only arrive here if precursor is active."));

  // Setup time integrator and get time step size
  double const start_time = std::min(application->precursor->get_parameters().start_time,
                                     application->main->get_parameters().start_time);

  // Set the same time step size for both time integrators
  solver_precursor.time_integrator->reset_time(start_time);
  solver_main.time_integrator->reset_time(start_time);
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  AssertThrow(application->precursor_is_active(),
              dealii::ExcMessage("One should only arrive here if precursor is active."));

  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_pre = std::numeric_limits<double>::max();
  double time_step_size     = std::numeric_limits<double>::max();

  // get time step sizes
  if(use_adaptive_time_stepping == true)
  {
    if(solver_precursor.time_integrator->get_time() >
       application->precursor->get_parameters().start_time - EPSILON)
    {
      time_step_size_pre = solver_precursor.time_integrator->get_time_step_size();
    }

    if(solver_main.time_integrator->get_time() >
       application->main->get_parameters().start_time - EPSILON)
    {
      time_step_size = solver_main.time_integrator->get_time_step_size();
    }
  }
  else
  {
    time_step_size_pre = solver_precursor.time_integrator->get_time_step_size();
    time_step_size     = solver_main.time_integrator->get_time_step_size();
  }

  // take the minimum
  time_step_size = std::min(time_step_size_pre, time_step_size);

  // decrease time_step in order to exactly hit end_time
  if(use_adaptive_time_stepping == false)
  {
    // assume that the precursor domain is the first to start and the last to end
    time_step_size =
      adjust_time_step_to_hit_end_time(application->precursor->get_parameters().start_time,
                                       application->precursor->get_parameters().end_time,
                                       time_step_size);

    pcout << std::endl
          << "Combined time step size for both domains: " << time_step_size << std::endl;
  }

  // set the time step size
  solver_precursor.time_integrator->set_current_time_step_size(time_step_size);
  solver_main.time_integrator->set_current_time_step_size(time_step_size);
}

// performs some additional parameter checks
template<int dim, typename Number>
void
Driver<dim, Number>::consistency_checks() const
{
  AssertThrow(application->precursor->get_parameters().ale_formulation == false,
              dealii::ExcMessage("not implemented."));
  AssertThrow(application->main->get_parameters().ale_formulation == false,
              dealii::ExcMessage("not implemented."));

  AssertThrow(application->precursor->get_parameters().calculation_of_time_step_size ==
                application->main->get_parameters().calculation_of_time_step_size,
              dealii::ExcMessage(
                "Type of time step calculation has to be the same for both domains."));

  AssertThrow(application->precursor->get_parameters().adaptive_time_stepping ==
                application->main->get_parameters().adaptive_time_stepping,
              dealii::ExcMessage(
                "Type of time step calculation has to be the same for both domains."));

  AssertThrow(application->precursor->get_parameters().solver_type == SolverType::Unsteady and
                application->main->get_parameters().solver_type == SolverType::Unsteady,
              dealii::ExcMessage("This is an unsteady solver. Check parameters."));

  // For the two-domain solver the parameter start_with_low_order has to be true.
  // This is due to the fact that the setup function of the time integrator initializes
  // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
  // start_with_low_order == false. However, the combined time step size
  // is not known at this point since the two domains have to first communicate with each other
  // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
  // inconsistencies is to preclude the case start_with_low_order == false.
  AssertThrow(application->precursor->get_parameters().start_with_low_order == true and
                application->main->get_parameters().start_with_low_order == true,
              dealii::ExcMessage("start_with_low_order has to be true for two-domain solver."));
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up incompressible Navier-Stokes solver:" << std::endl;

  AssertThrow(application->main.get(), dealii::ExcMessage("Domain main is uninitialized."));
  AssertThrow(application->precursor.get(),
              dealii::ExcMessage("Domain precursor is uninitialized."));

  // main domain
  application->main->setup(grid_main, mapping_main, {"Main"});

  // precursor domain
  if(application->precursor_is_active())
  {
    application->precursor->setup(grid_precursor, mapping_precursor, {"Precursor"});

    // make some additional checks (i.e. enforce constraints between main and precursor
    // parameters)
    consistency_checks();
  }

  // constant vs. adaptive time stepping
  use_adaptive_time_stepping = application->main->get_parameters().adaptive_time_stepping;

  // setup "solvers"
  solver_main.setup(application->main, grid_main, mapping_main, "main", mpi_comm, is_test);

  if(application->precursor_is_active())
  {
    solver_precursor.setup(
      application->precursor, grid_precursor, mapping_precursor, "precursor", mpi_comm, is_test);
  }

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  // time loop
  if(application->precursor_is_active())
  {
    set_start_time();

    synchronize_time_step_size();

    do
    {
      // advance one time step for precursor domain
      solver_precursor.time_integrator->advance_one_timestep();

      // Note that the coupling of both solvers via the inflow boundary conditions is
      // performed in the postprocessing step of the solver for the precursor domain,
      // overwriting the data global structures which are subsequently used by the
      // solver for the actual domain to evaluate the boundary conditions.

      // advance one time step for actual domain
      solver_main.time_integrator->advance_one_timestep();

      // Both domains have already calculated the new, adaptive time step size individually in
      // function advance_one_timestep(). Here, we have to synchronize the time step size for
      // both domains.
      if(use_adaptive_time_stepping == true)
      {
        synchronize_time_step_size();
      }
    } while(not(solver_precursor.time_integrator->finished()) or
            not(solver_main.time_integrator->finished()));
  }
  else
  {
    solver_main.time_integrator->timeloop();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  // Iterations
  pcout << std::endl
        << "Average number of iterations for incompressible Navier-Stokes solver:" << std::endl;

  pcout << std::endl << "Precursor:" << std::endl;

  solver_precursor.time_integrator->print_iterations();

  pcout << std::endl << "Main:" << std::endl;

  solver_main.time_integrator->print_iterations();

  // Wall times
  pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  timer_tree.insert({"Incompressible flow"}, total_time);

  timer_tree.insert({"Incompressible flow"},
                    solver_precursor.time_integrator->get_timings(),
                    "Timeloop precursor");

  timer_tree.insert({"Incompressible flow"},
                    solver_main.time_integrator->get_timings(),
                    "Timeloop main");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Computational costs in CPUh
  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG
