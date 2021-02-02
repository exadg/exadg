/*
 * driver_precursor.cpp
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/driver_precursor.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
DriverPrecursor<dim, Number>::DriverPrecursor(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    use_adaptive_time_stepping(false)
{
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const start_time = std::min(param_pre.start_time, param.start_time);

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
    if(time_integrator_pre->get_time() > param_pre.start_time - EPSILON)
      time_step_size_pre = time_integrator_pre->get_time_step_size();

    if(time_integrator->get_time() > param.start_time - EPSILON)
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
      adjust_time_step_to_hit_end_time(param_pre.start_time, param_pre.end_time, time_step_size);

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
                                    unsigned int const                                     degree,
                                    unsigned int const refine_space,
                                    bool const         is_test)
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

  application->set_input_parameters_precursor(param_pre);
  param_pre.check_input_parameters(pcout);
  param_pre.print(pcout, "List of input parameters for precursor domain:");

  application->set_input_parameters(param);
  param.check_input_parameters(pcout);
  param.print(pcout, "List of input parameters for actual domain:");

  AssertThrow(param_pre.ale_formulation == false, ExcMessage("not implemented."));
  AssertThrow(param.ale_formulation == false, ExcMessage("not implemented."));

  // triangulation
  if(param_pre.triangulation_type == TriangulationType::Distributed)
  {
    triangulation_pre.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param_pre.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation_pre.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  if(param.triangulation_type == TriangulationType::Distributed)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  // create grid
  application->create_grid_precursor(triangulation_pre, refine_space, periodic_faces_pre);
  application->create_grid(triangulation, refine_space, periodic_faces);

  print_grid_data(pcout, refine_space, *triangulation_pre);
  print_grid_data(pcout, refine_space, *triangulation);

  boundary_descriptor_velocity_pre.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_pre.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions_precursor(boundary_descriptor_velocity_pre,
                                                 boundary_descriptor_pressure_pre);
  verify_boundary_conditions(*boundary_descriptor_velocity_pre,
                             *triangulation_pre,
                             periodic_faces_pre);
  verify_boundary_conditions(*boundary_descriptor_pressure_pre,
                             *triangulation_pre,
                             periodic_faces_pre);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);
  verify_boundary_conditions(*boundary_descriptor_velocity, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_pressure, *triangulation, periodic_faces);

  field_functions_pre.reset(new FieldFunctions<dim>());
  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions_precursor(field_functions_pre);
  application->set_field_functions(field_functions);

  // constant vs. adaptive time stepping
  use_adaptive_time_stepping = param_pre.adaptive_time_stepping;

  AssertThrow(param_pre.calculation_of_time_step_size == param.calculation_of_time_step_size,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(param_pre.adaptive_time_stepping == param.adaptive_time_stepping,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(param_pre.solver_type == SolverType::Unsteady &&
                param.solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // mapping
  unsigned int const mapping_degree_pre = get_mapping_degree(param_pre.mapping, degree);
  mesh_pre.reset(new Mesh<dim>(mapping_degree_pre));

  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // initialize navier_stokes_operator_pre (precursor domain)
  if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled_pre.reset(new DGCoupled(*triangulation_pre,
                                                           mesh_pre->get_mapping(),
                                                           degree,
                                                           periodic_faces_pre,
                                                           boundary_descriptor_velocity_pre,
                                                           boundary_descriptor_pressure_pre,
                                                           field_functions_pre,
                                                           param_pre,
                                                           "fluid",
                                                           mpi_comm));

    navier_stokes_operator_pre = navier_stokes_operator_coupled_pre;
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting_pre.reset(
      new DGDualSplitting(*triangulation_pre,
                          mesh_pre->get_mapping(),
                          degree,
                          periodic_faces_pre,
                          boundary_descriptor_velocity_pre,
                          boundary_descriptor_pressure_pre,
                          field_functions_pre,
                          param_pre,
                          "fluid",
                          mpi_comm));

    navier_stokes_operator_pre = navier_stokes_operator_dual_splitting_pre;
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction_pre.reset(
      new DGPressureCorrection(*triangulation_pre,
                               mesh_pre->get_mapping(),
                               degree,
                               periodic_faces_pre,
                               boundary_descriptor_velocity_pre,
                               boundary_descriptor_pressure_pre,
                               field_functions_pre,
                               param_pre,
                               "fluid",
                               mpi_comm));

    navier_stokes_operator_pre = navier_stokes_operator_pressure_correction_pre;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize navier_stokes_operator (actual domain)
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled.reset(new DGCoupled(*triangulation,
                                                       mesh->get_mapping(),
                                                       degree,
                                                       periodic_faces,
                                                       boundary_descriptor_velocity,
                                                       boundary_descriptor_pressure,
                                                       field_functions,
                                                       param,
                                                       "fluid",
                                                       mpi_comm));

    navier_stokes_operator = navier_stokes_operator_coupled;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting.reset(new DGDualSplitting(*triangulation,
                                                                    mesh->get_mapping(),
                                                                    degree,
                                                                    periodic_faces,
                                                                    boundary_descriptor_velocity,
                                                                    boundary_descriptor_pressure,
                                                                    field_functions,
                                                                    param,
                                                                    "fluid",
                                                                    mpi_comm));

    navier_stokes_operator = navier_stokes_operator_dual_splitting;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction.reset(
      new DGPressureCorrection(*triangulation,
                               mesh->get_mapping(),
                               degree,
                               periodic_faces,
                               boundary_descriptor_velocity,
                               boundary_descriptor_pressure,
                               field_functions,
                               param,
                               "fluid",
                               mpi_comm));

    navier_stokes_operator = navier_stokes_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free precursor
  matrix_free_data_pre.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data_pre->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  if(param_pre.use_cell_based_face_loops)
  {
    auto tria =
      std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation_pre);
    Categorization::do_cell_based_loops(*tria, matrix_free_data_pre->data);
  }
  navier_stokes_operator_pre->fill_matrix_free_data(*matrix_free_data_pre);
  matrix_free_pre.reset(new MatrixFree<dim, Number>());
  matrix_free_pre->reinit(mesh_pre->get_mapping(),
                          matrix_free_data_pre->get_dof_handler_vector(),
                          matrix_free_data_pre->get_constraint_vector(),
                          matrix_free_data_pre->get_quadrature_vector(),
                          matrix_free_data_pre->data);

  // initialize matrix_free
  matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  if(param.use_cell_based_face_loops)
  {
    auto tria =
      std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation);
    Categorization::do_cell_based_loops(*tria, matrix_free_data->data);
  }
  navier_stokes_operator->fill_matrix_free_data(*matrix_free_data);
  matrix_free.reset(new MatrixFree<dim, Number>());
  matrix_free->reinit(mesh->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);


  // setup Navier-Stokes operator
  navier_stokes_operator_pre->setup(matrix_free_pre, matrix_free_data_pre);
  navier_stokes_operator->setup(matrix_free, matrix_free_data);

  // setup postprocessor
  postprocessor_pre = application->construct_postprocessor_precursor(degree, mpi_comm);
  postprocessor_pre->setup(*navier_stokes_operator_pre);

  postprocessor = application->construct_postprocessor(degree, mpi_comm);
  postprocessor->setup(*navier_stokes_operator);


  // Setup time integrator

  if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_pre.reset(new TimeIntCoupled(navier_stokes_operator_coupled_pre,
                                                 param_pre,
                                                 0 /* refine_time */,
                                                 mpi_comm,
                                                 not(is_test),
                                                 postprocessor_pre));
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_pre.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting_pre,
                                                       param_pre,
                                                       0 /* refine_time */,
                                                       mpi_comm,
                                                       not(is_test),
                                                       postprocessor_pre));
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pre.reset(
      new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction_pre,
                                    param_pre,
                                    0 /* refine_time */,
                                    mpi_comm,
                                    not(is_test),
                                    postprocessor_pre));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator.reset(new TimeIntCoupled(navier_stokes_operator_coupled,
                                             param,
                                             0 /* refine_time */,
                                             mpi_comm,
                                             not(is_test),
                                             postprocessor));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting,
                                                   param,
                                                   0 /* refine_time */,
                                                   mpi_comm,
                                                   not(is_test),
                                                   postprocessor));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator.reset(new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction,
                                                        param,
                                                        0 /* refine_time */,
                                                        mpi_comm,
                                                        not(is_test),
                                                        postprocessor));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }


  // For the two-domain solver the parameter start_with_low_order has to be true.
  // This is due to the fact that the setup function of the time integrator initializes
  // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
  // start_with_low_order == false. However, the combined time step size
  // is not known at this point since the two domains have to first communicate with each other
  // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
  // inconsistencies is to preclude the case start_with_low_order == false.
  AssertThrow(param_pre.start_with_low_order == true && param.start_with_low_order == true,
              ExcMessage("start_with_low_order has to be true for two-domain solver."));

  // setup time integrator before calling setup_solvers (this is necessary since the setup of the
  // solvers depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator_pre->setup(param_pre.restarted_simulation);
  time_integrator->setup(param.restarted_simulation);

  // setup solvers

  navier_stokes_operator_pre->setup_solvers(
    time_integrator_pre->get_scaling_factor_time_derivative_term(),
    time_integrator_pre->get_velocity());

  navier_stokes_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
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
DriverPrecursor<dim, Number>::print_performance_results(double const total_time,
                                                        bool const   is_test) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // Iterations
  this->pcout << std::endl
              << "Average number of iterations for incompressible Navier-Stokes solver:"
              << std::endl;

  this->pcout << std::endl << "Precursor:" << std::endl;

  time_integrator_pre->print_iterations();

  this->pcout << std::endl << "Main:" << std::endl;

  time_integrator->print_iterations();

  // Wall times
  this->pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  timer_tree.insert({"Incompressible flow"}, total_time);

  timer_tree.insert({"Incompressible flow"},
                    time_integrator_pre->get_timings(),
                    "Timeloop precursor");
  timer_tree.insert({"Incompressible flow"}, time_integrator->get_timings(), "Timeloop main");

  if(not(is_test))
  {
    pcout << std::endl << "Timings for level 1:" << std::endl;
    timer_tree.print_level(pcout, 1);

    pcout << std::endl << "Timings for level 2:" << std::endl;
    timer_tree.print_level(pcout, 2);

    // Computational costs in CPUh
    unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

    Utilities::MPI::MinMaxAvg total_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
    double const              total_time_avg  = total_time_data.avg;

    print_costs(pcout, total_time_avg, N_mpi_processes);
  }

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template class DriverPrecursor<2, float>;
template class DriverPrecursor<3, float>;

template class DriverPrecursor<2, double>;
template class DriverPrecursor<3, double>;

} // namespace IncNS
} // namespace ExaDG
