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
#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(std::string const &                           input_file,
                            MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);

  dealii::ParameterHandler prm;
  parameters.add_parameters(prm);
  prm.parse_input(input_file, "", true, true);

  structure = std::make_shared<WrapperStructure<dim, Number>>();
  fluid     = std::make_shared<WrapperFluid<dim, Number>>();

  partitioned_solver = std::make_shared<PartitionedSolver<dim, Number>>(parameters, mpi_comm);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

  // setup application
  {
    dealii::Timer timer_local;

    application->setup();

    timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
  }

  // setup structure
  {
    dealii::Timer timer_local;

    structure->setup(application->structure, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

  // setup fluid
  {
    dealii::Timer timer_local;

    fluid->setup(application->fluid, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  // setup interface coupling
  setup_interface_coupling();

  partitioned_solver->setup(fluid, structure);

  timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_interface_coupling()
{
  // structure to ALE
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> ALE ..." << std::endl;

    if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
      structure_to_ale->setup(fluid->ale_poisson_operator->get_container_interface_data(),
                              application->structure->get_boundary_descriptor()->neumann_cached_bc,
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              parameters.geometric_tolerance);
    }
    else if(application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
      structure_to_ale->setup(
        fluid->ale_elasticity_operator->get_container_interface_data_dirichlet(),
        application->structure->get_boundary_descriptor()->neumann_cached_bc,
        structure->pde_operator->get_dof_handler(),
        *application->structure->get_grid()->mapping,
        parameters.geometric_tolerance);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> ALE"}, timer_local.wall_time());
  }

  // structure to fluid
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> fluid ..." << std::endl;

    structure_to_fluid = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    structure_to_fluid->setup(fluid->pde_operator->get_container_interface_data(),
                              application->structure->get_boundary_descriptor()->neumann_cached_bc,
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> fluid"}, timer_local.wall_time());
  }

  // fluid to structure
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling fluid -> structure ..." << std::endl;

    fluid_to_structure = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    std::shared_ptr<dealii::Mapping<dim> const> mapping_fluid =
      get_dynamic_mapping<dim, Number>(application->fluid->get_grid(), fluid->ale_grid_motion);
    fluid_to_structure->setup(
      structure->pde_operator->get_container_interface_data_neumann(),
      application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc,
      fluid->pde_operator->get_dof_handler_u(),
      *mapping_fluid,
      parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling fluid -> structure"}, timer_local.wall_time());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // The fluid domain is the master that dictates the start time
  structure->time_integrator->reset_time(fluid->time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  // The fluid domain is the master that dictates the time step size
  structure->time_integrator->set_current_time_step_size(
    fluid->time_integrator->get_time_step_size());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_ale(VectorType const & displacement_structure) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  structure_to_ale->update_data(displacement_structure);
  timer_tree.insert({"FSI", "Coupling structure -> ALE"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_fluid(bool const extrapolate) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType velocity_structure;
  structure->pde_operator->initialize_dof_vector(velocity_structure);
  if(extrapolate)
    structure->time_integrator->extrapolate_velocity_to_np(velocity_structure);
  else
    velocity_structure = structure->time_integrator->get_velocity_np();

  structure_to_fluid->update_data(velocity_structure);

  timer_tree.insert({"FSI", "Coupling structure -> fluid"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_fluid_to_structure(bool const end_of_time_step) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType stress_fluid;
  fluid->pde_operator->initialize_vector_velocity(stress_fluid);
  // calculate fluid stress at fluid-structure interface
  if(end_of_time_step)
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
  }
  else
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity(),
                                               fluid->time_integrator->get_pressure());
  }

  stress_fluid *= -1.0;
  fluid_to_structure->update_data(stress_fluid);

  timer_tree.insert({"FSI", "Coupling fluid -> structure"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::apply_dirichlet_neumann_scheme(VectorType &       d_tilde,
                                                    VectorType const & d,
                                                    unsigned int       iteration) const
{
  coupling_structure_to_ale(d);

  // move the fluid mesh and update dependent data structures
  fluid->solve_ale(application->fluid, is_test);

  // update velocity boundary condition for fluid
  coupling_structure_to_fluid(iteration == 0);

  // solve fluid problem
  fluid->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0);

  // update stress boundary condition for solid
  coupling_fluid_to_structure(/* end_of_time_step = */ true);

  // solve structural problem
  structure->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0);

  d_tilde = structure->time_integrator->get_displacement_np();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // compute initial acceleration for structural problem
  {
    // update stress boundary condition for solid at time t_n (not t_{n+1})
    coupling_fluid_to_structure(/* end_of_time_step = */ false);
    structure->time_integrator->compute_initial_acceleration(
      application->structure->get_parameters().restarted_simulation);
  }

  // The fluid domain is the master that dictates when the time loop is finished
  while(!fluid->time_integrator->finished())
  {
    // pre-solve
    fluid->time_integrator->advance_one_timestep_pre_solve(true);
    structure->time_integrator->advance_one_timestep_pre_solve(false);

    // solve (using strongly-coupled partitioned scheme)
    auto const lambda_dirichlet_neumann =
      [&](VectorType & d_tilde, VectorType const & d, unsigned int k) {
        apply_dirichlet_neumann_scheme(d_tilde, d, k);
      };
    partitioned_solver->solve(lambda_dirichlet_neumann);

    // post-solve
    fluid->time_integrator->advance_one_timestep_post_solve();
    structure->time_integrator->advance_one_timestep_post_solve();

    if(application->fluid->get_parameters().adaptive_time_stepping)
      synchronize_time_step_size();
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

  pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  partitioned_solver->print_iterations(pcout);

  pcout << std::endl << "Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "ALE:" << std::endl;
  fluid->ale_grid_motion->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure->time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, fluid->get_timings_ale());
  timer_tree.insert({"FSI"}, structure->time_integrator->get_timings(), "Structure");
  timer_tree.insert({"FSI"}, partitioned_solver->get_timings());

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index DoFs =
    fluid->pde_operator->get_number_of_dofs() + structure->pde_operator->get_number_of_dofs();

  if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
  {
    DoFs += fluid->pde_operator->get_number_of_dofs();
  }
  else if(application->fluid->get_parameters().mesh_movement_type ==
          IncNS::MeshMovementType::Elasticity)
  {
    DoFs += fluid->ale_elasticity_operator->get_number_of_dofs();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  unsigned int N_time_steps = fluid->time_integrator->get_number_of_time_steps();

  print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FSI
} // namespace ExaDG
