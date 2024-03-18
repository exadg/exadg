/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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
#include <exadg/aero_acoustic/driver.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace AeroAcoustic
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    application(app),
    acoustic(std::make_shared<SolverAcoustic<dim, Number>>()),
    fluid(std::make_shared<SolverFluid<dim, Number>>()),
    time_solvers_side_by_side(std::numeric_limits<double>::min())
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up aero-acoustic solver:" << std::endl;

  // setup application
  application->setup();

  // setup acoustic solver
  {
    dealii::Timer timer_local;

    acoustic->setup(application->acoustic, mpi_comm, is_test);

    timer_tree.insert({"AeroAcoustic", "Setup", "Acoustic"}, timer_local.wall_time());
  }

  // setup fluid solver
  {
    dealii::Timer timer_local;

    fluid->setup(application->fluid, mpi_comm, is_test);

    timer_tree.insert({"AeroAcoustic", "Setup", "Fluid"}, timer_local.wall_time());
  }

  setup_volume_coupling();

  timer_tree.insert({"AeroAcoustic", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_volume_coupling()
{
  // fluid to structure
  {
    dealii::Timer timer_local;

    pcout << std::endl << "Setup volume coupling fluid -> acoustic ..." << std::endl;

    volume_coupling.setup(application->parameters, acoustic, fluid);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"AeroAcoustic", "Setup", "Coupling fluid -> acoustic"},
                      timer_local.wall_time());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  AssertThrow(fluid->time_integrator->get_time() - 1e-12 < acoustic->time_integrator->get_time(),
              dealii::ExcMessage(
                "Acoustic simulation can not be started before fluid simulation."));

  acoustic->time_integrator->reset_time(fluid->time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::couple_fluid_to_acoustic()
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  volume_coupling.fluid_to_acoustic();

  timer_tree.insert({"AeroAcoustic", "Coupling fluid -> acoustic"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  std::pair<bool, dealii::Timer> timer = std::make_pair(false, dealii::Timer());

  set_start_time();

  AssertThrow(std::abs(application->fluid->get_parameters().end_time -
                       application->acoustic->get_parameters().end_time) < 1.0e-12,
              dealii::ExcMessage("Acoustic and fluid simulation need the same end time."));

  while(not acoustic->time_integrator->finished())
  {
    if(timer.first == false && acoustic->time_integrator->started())
    {
      timer.first = true;
      timer.second.restart();
    }

    // To check if acoustics starts during the following sub-stepping sweep we
    // can not simply check acoustic->time_integrator->started(). Instead we
    // compute this information as follows:
    const bool acoustic_starts_during_sub_stepping =
      fluid->time_integrator->get_next_time() + fluid->time_integrator->get_time_step_size() >
      application->acoustic->get_parameters().start_time;

    // The acoustic simulation uses explicit time-stepping while the fluid solver
    // uses implicit time-stepping. Therefore, we advance the acoustic solver to
    // t^(n+1) first and directly use the result in the fluid solver.
    if(acoustic_starts_during_sub_stepping)
      couple_fluid_to_acoustic();
    acoustic->advance_multiple_timesteps(fluid->time_integrator->get_time_step_size());

    // We can not simply check acoustic->time_integrator->started() since acoustic might start
    // during a sub-stepping sweep. To check if the acoustic might be started during the next
    // sub-stepping BEFORE performing the current fluid time step we have to check if dt_{n+2}
    // is larger than the acoustic start time. We need this information BEFORE the fluid
    // time-step since we have to know if we have to compute dp/dt.
    const bool acoustic_might_start =
      fluid->time_integrator->get_next_time() + fluid->max_next_time_step_size() >
      application->acoustic->get_parameters().start_time;

    fluid->advance_one_timestep_and_compute_pressure_time_derivative(acoustic_might_start);
  }

  time_solvers_side_by_side = timer.second.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Performance results for aero-acoustic solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "Average number of sub-time steps Acoustic:" << std::endl;
  pcout << "Adams-Bashforth-Moulton    " << acoustic->get_average_sub_time_steps() << std::endl;

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"AeroAcoustic"}, total_time);

  timer_tree.insert({"AeroAcoustic"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"AeroAcoustic"}, acoustic->time_integrator->get_timings(), "Acoustic");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core (during the time both
  // solvers ran side by side)
  dealii::Utilities::MPI::MinMaxAvg time_solvers_side_by_side_data =
    dealii::Utilities::MPI::min_max_avg(time_solvers_side_by_side, mpi_comm);
  double const time_solvers_side_by_side_avg = time_solvers_side_by_side_data.avg;

  dealii::types::global_dof_index const DoFs =
    fluid->pde_operator->get_number_of_dofs() +
    acoustic->get_average_sub_time_steps() * acoustic->pde_operator->get_number_of_dofs();

  unsigned int const N_time_steps    = acoustic->get_number_of_global_time_steps();
  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  pcout << std::endl << "Throughput while both solvers ran side by side:";
  print_throughput_unsteady(
    pcout, DoFs, time_solvers_side_by_side_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << print_horizontal_line() << std::endl << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace AeroAcoustic
} // namespace ExaDG
