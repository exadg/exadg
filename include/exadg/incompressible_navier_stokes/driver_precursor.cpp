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

#include <exadg/incompressible_navier_stokes/driver_precursor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
DriverPrecursor<dim, Number>::DriverPrecursor(MPI_Comm const & comm, bool const is_test)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    use_adaptive_time_stepping(false)
{
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const start_time = std::min(application->get_parameters_precursor().start_time,
                                     application->get_parameters().start_time);

  // Set the same time step size for both time integrators
  time_integrator_pre->reset_time(start_time);
  time_integrator->reset_time(start_time);
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_pre = std::numeric_limits<double>::max();
  double time_step_size     = std::numeric_limits<double>::max();

  // get time step sizes
  if(use_adaptive_time_stepping == true)
  {
    if(time_integrator_pre->get_time() >
       application->get_parameters_precursor().start_time - EPSILON)
      time_step_size_pre = time_integrator_pre->get_time_step_size();

    if(time_integrator->get_time() > application->get_parameters().start_time - EPSILON)
      time_step_size = time_integrator->get_time_step_size();
  }
  else
  {
    time_step_size_pre = time_integrator_pre->get_time_step_size();
    time_step_size     = time_integrator->get_time_step_size();
  }

  // take the minimum
  time_step_size = std::min(time_step_size_pre, time_step_size);

  // decrease time_step in order to exactly hit end_time
  if(use_adaptive_time_stepping == false)
  {
    // assume that the precursor domain is the first to start and the last to end
    time_step_size =
      adjust_time_step_to_hit_end_time(application->get_parameters_precursor().start_time,
                                       application->get_parameters_precursor().end_time,
                                       time_step_size);

    pcout << std::endl
          << "Combined time step size for both domains: " << time_step_size << std::endl;
  }

  // set the time step size
  time_integrator_pre->set_current_time_step_size(time_step_size);
  time_integrator->set_current_time_step_size(time_step_size);
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::setup(std::shared_ptr<ApplicationBasePrecursor<dim, Number>> app,
                                    unsigned int const degree_velocity,
                                    unsigned int const refine_space)
{
  Timer timer;
  timer.restart();

  print_exadg_header(pcout);
  pcout << "Setting up incompressible Navier-Stokes solver:" << std::endl;

  if(not(is_test))
  {
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_parameters_precursor(degree_velocity);
  application->get_parameters_precursor().check(pcout);
  application->get_parameters_precursor().print(pcout, "List of parameters for precursor domain:");

  application->set_parameters(degree_velocity);
  application->get_parameters().check(pcout);
  application->get_parameters().print(pcout, "List of parameters for actual domain:");

  AssertThrow(application->get_parameters_precursor().ale_formulation == false,
              ExcMessage("not implemented."));
  AssertThrow(application->get_parameters().ale_formulation == false,
              ExcMessage("not implemented."));

  // grid_pre
  GridData grid_data_pre;
  grid_data_pre.triangulation_type = application->get_parameters_precursor().triangulation_type;
  grid_data_pre.n_refine_global    = refine_space;
  grid_data_pre.mapping_degree =
    get_mapping_degree(application->get_parameters_precursor().mapping,
                       application->get_parameters_precursor().degree_u);

  grid_pre = application->create_grid_precursor(grid_data_pre);
  print_grid_info(pcout, *grid_pre);

  // grid
  GridData grid_data;
  grid_data.triangulation_type = application->get_parameters().triangulation_type;
  grid_data.n_refine_global    = refine_space;
  grid_data.mapping_degree     = get_mapping_degree(application->get_parameters().mapping,
                                                application->get_parameters().degree_u);

  grid = application->create_grid(grid_data);
  print_grid_info(pcout, *grid);

  application->set_boundary_descriptor_precursor();
  IncNS::verify_boundary_conditions<dim>(application->get_boundary_descriptor_precursor(),
                                         *grid_pre);

  application->set_boundary_descriptor();
  IncNS::verify_boundary_conditions<dim>(application->get_boundary_descriptor(), *grid);

  application->set_field_functions_precursor();
  application->set_field_functions();

  // constant vs. adaptive time stepping
  use_adaptive_time_stepping = application->get_parameters_precursor().adaptive_time_stepping;

  AssertThrow(application->get_parameters_precursor().calculation_of_time_step_size ==
                application->get_parameters().calculation_of_time_step_size,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(application->get_parameters_precursor().adaptive_time_stepping ==
                application->get_parameters().adaptive_time_stepping,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(application->get_parameters_precursor().solver_type == SolverType::Unsteady &&
                application->get_parameters().solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check parameters."));

  // initialize pde_operator_pre (precursor domain)
  pde_operator = create_operator<dim, Number>(grid_pre,
                                              application->get_boundary_descriptor_precursor(),
                                              application->get_field_functions_precursor(),
                                              application->get_parameters_precursor(),
                                              "fluid",
                                              mpi_comm);

  // initialize operator_base (actual domain)
  pde_operator = create_operator<dim, Number>(grid,
                                              application->get_boundary_descriptor(),
                                              application->get_field_functions(),
                                              application->get_parameters(),
                                              "fluid",
                                              mpi_comm);


  // initialize matrix_free precursor
  matrix_free_data_pre = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data_pre->append(pde_operator_pre);

  matrix_free_pre = std::make_shared<MatrixFree<dim, Number>>();
  if(application->get_parameters_precursor().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid_pre->triangulation, matrix_free_data_pre->data);
  matrix_free_pre->reinit(*grid_pre->mapping,
                          matrix_free_data_pre->get_dof_handler_vector(),
                          matrix_free_data_pre->get_constraint_vector(),
                          matrix_free_data_pre->get_quadrature_vector(),
                          matrix_free_data_pre->data);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  if(application->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);
  matrix_free->reinit(*grid->mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);


  // setup Navier-Stokes operator
  pde_operator_pre->setup(matrix_free_pre, matrix_free_data_pre);
  pde_operator->setup(matrix_free, matrix_free_data);

  // setup postprocessor
  postprocessor_pre = application->create_postprocessor_precursor();
  postprocessor_pre->setup(*pde_operator_pre);

  postprocessor = application->create_postprocessor();
  postprocessor->setup(*pde_operator);


  // Setup time integrator
  time_integrator_pre = create_time_integrator<dim, Number>(pde_operator_pre,
                                                            application->get_parameters_precursor(),
                                                            0 /* refine_time */,
                                                            mpi_comm,
                                                            is_test,
                                                            postprocessor_pre);

  time_integrator = create_time_integrator<dim, Number>(pde_operator,
                                                        application->get_parameters(),
                                                        0 /* refine_time */,
                                                        mpi_comm,
                                                        is_test,
                                                        postprocessor);


  // For the two-domain solver the parameter start_with_low_order has to be true.
  // This is due to the fact that the setup function of the time integrator initializes
  // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
  // start_with_low_order == false. However, the combined time step size
  // is not known at this point since the two domains have to first communicate with each other
  // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
  // inconsistencies is to preclude the case start_with_low_order == false.
  AssertThrow(application->get_parameters_precursor().start_with_low_order == true &&
                application->get_parameters().start_with_low_order == true,
              ExcMessage("start_with_low_order has to be true for two-domain solver."));

  // setup time integrator before calling setup_solvers (this is necessary since the setup of the
  // solvers depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator_pre->setup(application->get_parameters_precursor().restarted_simulation);
  time_integrator->setup(application->get_parameters().restarted_simulation);

  // setup solvers

  pde_operator_pre->setup_solvers(time_integrator_pre->get_scaling_factor_time_derivative_term(),
                                  time_integrator_pre->get_velocity());

  pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                              time_integrator->get_velocity());

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // time loop
  do
  {
    // advance one time step for precursor domain
    time_integrator_pre->advance_one_timestep();

    // Note that the coupling of both solvers via the inflow boundary conditions is
    // performed in the postprocessing step of the solver for the precursor domain,
    // overwriting the data global structures which are subsequently used by the
    // solver for the actual domain to evaluate the boundary conditions.

    // advance one time step for actual domain
    time_integrator->advance_one_timestep();

    // Both domains have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size for
    // both domains.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();
  } while(!time_integrator_pre->finished() || !time_integrator->finished());
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  // Iterations
  pcout << std::endl
        << "Average number of iterations for incompressible Navier-Stokes solver:" << std::endl;

  pcout << std::endl << "Precursor:" << std::endl;

  time_integrator_pre->print_iterations();

  pcout << std::endl << "Main:" << std::endl;

  time_integrator->print_iterations();

  // Wall times
  pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  timer_tree.insert({"Incompressible flow"}, total_time);

  timer_tree.insert({"Incompressible flow"},
                    time_integrator_pre->get_timings(),
                    "Timeloop precursor");

  timer_tree.insert({"Incompressible flow"}, time_integrator->get_timings(), "Timeloop main");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Computational costs in CPUh
  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg total_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              total_time_avg  = total_time_data.avg;

  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class DriverPrecursor<2, float>;
template class DriverPrecursor<3, float>;

template class DriverPrecursor<2, double>;
template class DriverPrecursor<3, double>;

} // namespace IncNS
} // namespace ExaDG
