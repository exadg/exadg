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

  application->setup(grid, mapping, multigrid_mappings);

  bool const ale = application->get_parameters().ale_formulation;

  if(ale) // moving mesh
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion =
      application->create_mesh_movement_function();
    ale_mapping = std::make_shared<DeformedMappingFunction<dim, Number>>(
      mapping,
      application->get_parameters().degree,
      *grid->triangulation,
      mesh_motion,
      application->get_parameters().start_time);

    ale_multigrid_mappings = std::make_shared<MultigridMappings<dim, Number>>(
      ale_mapping, application->get_parameters().mapping_degree_coarse_grids);

    helpers_ale = std::make_shared<HelpersALE<dim, Number>>();

    helpers_ale->move_grid = [&](double const & time) {
      ale_mapping->update(time,
                          false /* print_solver_info */,
                          time_integrator->get_number_of_time_steps());
    };

    helpers_ale->update_pde_operator_after_grid_motion = [&]() {
      pde_operator->update_after_grid_motion(true);
    };

    helpers_ale->fill_grid_coordinates_vector = [&](VectorType & grid_coordinates,
                                                    dealii::DoFHandler<dim> const & dof_handler) {
      ale_mapping->fill_grid_coordinates_vector(grid_coordinates, dof_handler);
    };
  }

  // initialize convection-diffusion operator
  pde_operator =
    std::make_shared<Operator<dim, Number>>(grid,
                                            ale ? ale_mapping->get_mapping() : mapping,
                                            ale ? ale_multigrid_mappings : multigrid_mappings,
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
Driver<dim, Number>::mark_cells_coarsening_and_refinement(dealii::Triangulation<dim> & tria,
                                                          VectorType const & solution) const
{
  mark_cells_kelly_error_estimator(tria,
                                   pde_operator->get_dof_handler(),
                                   pde_operator->get_constraints(),
                                   *pde_operator->get_mapping(),
                                   solution,
                                   application->get_parameters().degree +
                                     1 /* n_face_quadrature_points */,
                                   application->get_parameters().amr_data);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_after_coarsening_and_refinement()
{
  // Update mapping
  AssertThrow(ale_mapping.get() == 0,
              dealii::ExcMessage(
                "Combination of adaptive mesh refinement and ALE not implemented."));

  std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache =
    std::dynamic_pointer_cast<dealii::MappingQCache<dim>>(mapping);
  AssertThrow(
    mapping_q_cache.get() == 0,
    dealii::ExcMessage(
      "Combination of adaptive mesh refinement and dealii::MappingQCache not implemented."));

  pde_operator->setup_after_coarsening_and_refinement();

  postprocessor->setup_after_coarsening_and_refinement();
}

template<int dim, typename Number>
void
Driver<dim, Number>::do_adaptive_refinement()
{
  limit_coarsening_and_refinement(*grid->triangulation, application->get_parameters().amr_data);

  if(any_cells_flagged_for_coarsening_or_refinement(*grid->triangulation))
  {
    grid->triangulation->prepare_coarsening_and_refinement();

    if(application->get_parameters().problem_type == ProblemType::Unsteady)
    {
      time_integrator->prepare_coarsening_and_refinement();
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    grid->triangulation->execute_coarsening_and_refinement();

    if(application->get_parameters().involves_h_multigrid())
    {
      GridUtilities::create_coarse_triangulations_after_coarsening_and_refinement(
        *grid->triangulation,
        grid->periodic_face_pairs,
        grid->coarse_triangulations,
        grid->coarse_periodic_face_pairs,
        application->get_parameters().grid,
        application->get_parameters().amr_data.preserve_boundary_cells);
    }

    setup_after_coarsening_and_refinement();

    if(application->get_parameters().problem_type == ProblemType::Unsteady)
    {
      time_integrator->interpolate_after_coarsening_and_refinement();
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    if(application->get_parameters().enable_adaptivity)
    {
      do
      {
        time_integrator->advance_one_timestep_pre_solve(true);

        time_integrator->advance_one_timestep_solve();

        // Adapt the mesh before post_solve(), in order to recalculate the
        // time step size based on the new mesh.
        if(trigger_coarsening_and_refinement_now(
             application->get_parameters().amr_data.trigger_every_n_time_steps,
             time_integrator->get_number_of_time_steps()))
        {
          // AMR is only implemented for implicit timestepping.
          std::shared_ptr<TimeIntBDF<dim, Number>> bdf_time_integrator =
            std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);

          AssertThrow(bdf_time_integrator.get(),
                      dealii::ExcMessage("Adaptive mesh refinement only implemented"
                                         " for implicit time integration."));

          mark_cells_coarsening_and_refinement(*grid->triangulation,
                                               bdf_time_integrator->get_solution_np());

          do_adaptive_refinement();
        }

        time_integrator->advance_one_timestep_post_solve();
      } while(not(time_integrator->finished()));
    }
    else
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
