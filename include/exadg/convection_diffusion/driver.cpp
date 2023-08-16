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

// likwid
#ifdef EXADG_WITH_LIKWID
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/convection_diffusion/driver.h>
#include <exadg/convection_diffusion/time_integration/create_time_integrator.h>
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/operators/throughput_parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test,
                            bool const                                    is_throughput_study)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    is_throughput_study(is_throughput_study),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up scalar convection-diffusion solver:" << std::endl;

  application->setup();

  if(application->get_parameters().ale_formulation) // moving mesh
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion =
      application->create_mesh_movement_function();
    ale_mapping = std::make_shared<DeformedMappingFunction<dim, Number>>(
      application->get_mapping(),
      application->get_parameters().degree,
      *application->get_grid()->triangulation,
      mesh_motion,
      application->get_parameters().start_time);

    helpers_ale = std::make_shared<HelpersALE<Number>>();

    helpers_ale->move_grid = [&](double const & time) {
      ale_mapping->update(time,
                          false /* print_solver_info */,
                          time_integrator->get_number_of_time_steps());
    };

    helpers_ale->update_pde_operator_after_grid_motion = [&]() {
      pde_operator->update_after_grid_motion(true);
    };
  }

  std::shared_ptr<dealii::Mapping<dim> const> dynamic_mapping =
    get_dynamic_mapping<dim, Number>(application->get_mapping(), ale_mapping);

  // initialize convection-diffusion operator
  pde_operator = std::make_shared<Operator<dim, Number>>(application->get_grid(),
                                                         dynamic_mapping,
                                                         application->get_boundary_descriptor(),
                                                         application->get_field_functions(),
                                                         application->get_parameters(),
                                                         "scalar",
                                                         mpi_comm);

  // setup convection-diffusion operator
  pde_operator->setup();

  if(not is_throughput_study)
  {
    // initialize postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // initialize time integrator or driver for steady problems
    if(application->get_parameters().problem_type == ProblemType::Unsteady)
    {
      time_integrator = create_time_integrator<dim, Number>(
        pde_operator, helpers_ale, postprocessor, application->get_parameters(), mpi_comm, is_test);

      time_integrator->setup(application->get_parameters().restarted_simulation);
    }
    else if(application->get_parameters().problem_type == ProblemType::Steady)
    {
      driver_steady = std::make_shared<DriverSteadyProblems<Number>>(
        pde_operator, postprocessor, application->get_parameters(), mpi_comm, is_test);

      driver_steady->setup();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented"));
    }

    // setup solvers in case of BDF time integration or steady problems
    typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
    VectorType const *                                         velocity_ptr = nullptr;
    VectorType                                                 velocity;

    if(application->get_parameters().problem_type == ProblemType::Unsteady)
    {
      if(application->get_parameters().temporal_discretization == TemporalDiscretization::BDF)
      {
        std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_bdf =
          std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);

        if(application->get_parameters().get_type_velocity_field() == TypeVelocityField::DoFVector)
        {
          pde_operator->initialize_dof_vector_velocity(velocity);
          pde_operator->interpolate_velocity(velocity, time_integrator->get_time());
          velocity_ptr = &velocity;
        }

        pde_operator->setup_solver(time_integrator_bdf->get_scaling_factor_time_derivative_term(),
                                   velocity_ptr);
      }
      else
      {
        AssertThrow(application->get_parameters().temporal_discretization ==
                      TemporalDiscretization::ExplRK,
                    dealii::ExcMessage("Not implemented."));
      }
    }
    else if(application->get_parameters().problem_type == ProblemType::Steady)
    {
      if(application->get_parameters().get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        pde_operator->initialize_dof_vector_velocity(velocity);
        pde_operator->interpolate_velocity(velocity, 0.0 /* time */);
        velocity_ptr = &velocity;
      }

      pde_operator->setup_solver(1.0 /* scaling_factor_time_derivative_term */, velocity_ptr);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented"));
    }
  }

  timer_tree.insert({"Convection-diffusion", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  // move the mesh and update dependent data structures
  helpers_ale->move_grid(time_integrator->get_next_time());

  helpers_ale->update_pde_operator_after_grid_motion();

  std::shared_ptr<TimeIntBDF<dim, Number>> time_int_bdf =
    std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);
  time_int_bdf->ale_update();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    if(application->get_parameters().ale_formulation == true)
    {
      do
      {
        time_integrator->advance_one_timestep_pre_solve(true);

        ale_update();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      } while(not(time_integrator->finished()));
    }
    else
    {
      time_integrator->timeloop();
    }
  }
  else if(application->get_parameters().problem_type == ProblemType::Steady)
  {
    driver_steady->solve();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented"));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for convection-diffusion solver:" << std::endl;

  // Averaged number of iterations are only relevant for BDF time integrator
  if(application->get_parameters().problem_type == ProblemType::Unsteady and
     application->get_parameters().temporal_discretization == TemporalDiscretization::BDF)
  {
    this->pcout << std::endl << "Average number of iterations:" << std::endl;

    std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_bdf =
      std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);
    time_integrator_bdf->print_iterations();
  }

  // wall times
  timer_tree.insert({"Convection-diffusion"}, total_time);

  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    if(application->get_parameters().temporal_discretization == TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<TimeIntExplRK<Number>> time_integrator_rk =
        std::dynamic_pointer_cast<TimeIntExplRK<Number>>(time_integrator);
      timer_tree.insert({"Convection-diffusion"}, time_integrator_rk->get_timings());
    }
    else if(application->get_parameters().temporal_discretization == TemporalDiscretization::BDF)
    {
      std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);
      timer_tree.insert({"Convection-diffusion"}, time_integrator_bdf->get_timings());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
  else
  {
    timer_tree.insert({"Convection-diffusion"}, driver_steady->get_timings());
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index const DoFs = pde_operator->get_number_of_dofs();
  unsigned int N_mpi_processes               = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg overall_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const overall_time_avg = overall_time_data.avg;

  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    unsigned int N_time_steps = this->time_integrator->get_number_of_time_steps();
    print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);
  }
  else
  {
    print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);
  }

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, dealii::types::global_dof_index, double>
Driver<dim, Number>::apply_operator(OperatorType const & operator_type,
                                    unsigned int const   n_repetitions_inner,
                                    unsigned int const   n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  dealii::LinearAlgebra::distributed::Vector<Number> dst, src;

  pde_operator->initialize_dof_vector(src);
  src = 1.0;
  pde_operator->initialize_dof_vector(dst);

  dealii::LinearAlgebra::distributed::Vector<Number> velocity;
  if(application->get_parameters().convective_problem())
  {
    if(application->get_parameters().get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      pde_operator->initialize_dof_vector_velocity(velocity);
      velocity = 1.0;
    }
  }

  if(operator_type == OperatorType::ConvectiveOperator)
    pde_operator->update_convective_term(1.0 /* time */, &velocity);
  else if(operator_type == OperatorType::MassConvectionDiffusionOperator)
    pde_operator->update_conv_diff_operator(1.0 /* time */,
                                            1.0 /* scaling_factor_mass */,
                                            &velocity);

  const std::function<void(void)> operator_evaluation = [&](void) {
    if(operator_type == OperatorType::MassOperator)
      pde_operator->apply_mass_operator(dst, src);
    else if(operator_type == OperatorType::ConvectiveOperator)
      pde_operator->apply_convective_term(dst, src);
    else if(operator_type == OperatorType::DiffusiveOperator)
      pde_operator->apply_diffusive_term(dst, src);
    else if(operator_type == OperatorType::MassConvectionDiffusionOperator)
      pde_operator->apply_conv_diff_operator(dst, src);
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(operator_evaluation,
                                                            application->get_parameters().degree,
                                                            n_repetitions_inner,
                                                            n_repetitions_outer,
                                                            mpi_comm);

  // calculate throughput
  dealii::types::global_dof_index const dofs = pde_operator->get_number_of_dofs();

  double const throughput = (double)dofs / wall_time;

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  if(not(is_test))
  {
    // clang-format off
    pcout << std::endl
          << std::scientific << std::setprecision(4)
          << "DoFs/sec:        " << throughput << std::endl
          << "DoFs/(sec*core): " << throughput/(double)N_mpi_processes << std::endl;
    // clang-format on
  }

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, dealii::types::global_dof_index, double>(
    application->get_parameters().degree, dofs, throughput);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
