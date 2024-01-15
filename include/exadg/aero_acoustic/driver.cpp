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
    fluid(std::make_shared<SolverFluid<dim, Number>>())
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
Driver<dim, Number>::synchronize_time_step_size() const
{
  // TODO: to avoid sophisticated methods for the minimal solver we simply
  // use the smallest time step.
  double const dt = std::min(fluid->time_integrator->get_time_step_size(),
                             acoustic->time_integrator->get_time_step_size());

  acoustic->time_integrator->set_current_time_step_size(dt);
  fluid->time_integrator->set_current_time_step_size(dt);
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
  set_start_time();

  synchronize_time_step_size();

  AssertThrow(std::abs(application->fluid->get_parameters().end_time -
                       application->acoustic->get_parameters().end_time) < 1.0e-12,
              dealii::ExcMessage("Acoustic and fluid simulation need the same end time."));

  while(not acoustic->time_integrator->finished())
  {
    // The acoustic simulation uses explicit time-stepping while the fluid solver
    // uses implicit time-stepping. Therefore, we advance the acoustic solver to
    // t^(n+1) first and directly use the result in the fluid solver.
    if(acoustic->time_integrator->started())
      couple_fluid_to_acoustic();
    acoustic->time_integrator->advance_one_timestep();

    fluid->advance_one_timestep_with_pressure_time_derivative(acoustic->time_integrator->started());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Performance results for aero-acoustic solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "Acoustic:" << std::endl;
  acoustic->time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"AeroAcoustic"}, total_time);

  timer_tree.insert({"AeroAcoustic"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"AeroAcoustic"}, acoustic->time_integrator->get_timings(), "Acoustic");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;


  // TODO: Print throughput statistics is not as easy since the number
  // of fluid and acoustic timesteps differ. This has to be implemented
  // once we extend the aerouacoustic solver.

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << print_horizontal_line() << std::endl << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace AeroAcoustic
} // namespace ExaDG
