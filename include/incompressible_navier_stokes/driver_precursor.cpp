/*
 * driver_precursor.cpp
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#include "driver_precursor.h"

namespace IncNS
{
template<int dim, typename Number>
DriverPrecursor<dim, Number>::DriverPrecursor(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    use_adaptive_time_stepping(false),
    overall_time(0.0),
    setup_time(0.0)
{
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                     based on a matrix-free implementation                       " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
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
                                    unsigned int const &                                   degree,
                                    unsigned int const & refine_space,
                                    unsigned int const & refine_time)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters_precursor(param_pre);
  // some parameters have to be overwritten
  param_pre.degree_u       = degree;
  param_pre.h_refinements  = refine_space;
  param_pre.dt_refinements = refine_time;

  param_pre.check_input_parameters(pcout);
  param_pre.print(pcout, "List of input parameters for precursor domain:");

  application->set_input_parameters(param);
  // some parameters have to be overwritten
  param.degree_u       = degree;
  param.h_refinements  = refine_space;
  param.dt_refinements = refine_time;

  param.check_input_parameters(pcout);
  param.print(pcout, "List of input parameters for actual domain:");

  AssertThrow(param_pre.dim == param.dim, ExcMessage("Invalid parameters."));

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
  application->create_grid_precursor(triangulation_pre,
                                     param_pre.h_refinements,
                                     periodic_faces_pre);
  application->create_grid(triangulation, param.h_refinements, periodic_faces);

  print_grid_data(
    pcout, param_pre.h_refinements, *triangulation_pre, param.h_refinements, *triangulation);

  boundary_descriptor_velocity_pre.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_pre.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions_precursor(boundary_descriptor_velocity_pre,
                                                 boundary_descriptor_pressure_pre);
  verify_boundary_conditions(*boundary_descriptor_velocity_pre,
                             *triangulation_pre,
                             periodic_faces_pre);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);
  verify_boundary_conditions(*boundary_descriptor_velocity, *triangulation, periodic_faces);

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
  unsigned int const mapping_degree_pre = get_mapping_degree(param_pre.mapping, param_pre.degree_u);
  mesh_pre.reset(new Mesh<dim>(mapping_degree_pre));

  unsigned int const mapping_degree = get_mapping_degree(param.mapping, param.degree_u);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // initialize navier_stokes_operator_pre (precursor domain)
  if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled_pre.reset(new DGCoupled(*triangulation_pre,
                                                           mesh_pre->get_mapping(),
                                                           periodic_faces_pre,
                                                           boundary_descriptor_velocity_pre,
                                                           boundary_descriptor_pressure_pre,
                                                           field_functions_pre,
                                                           param_pre,
                                                           mpi_comm));

    navier_stokes_operator_pre = navier_stokes_operator_coupled_pre;
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting_pre.reset(
      new DGDualSplitting(*triangulation_pre,
                          mesh_pre->get_mapping(),
                          periodic_faces_pre,
                          boundary_descriptor_velocity_pre,
                          boundary_descriptor_pressure_pre,
                          field_functions_pre,
                          param_pre,
                          mpi_comm));

    navier_stokes_operator_pre = navier_stokes_operator_dual_splitting_pre;
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction_pre.reset(
      new DGPressureCorrection(*triangulation_pre,
                               mesh_pre->get_mapping(),
                               periodic_faces_pre,
                               boundary_descriptor_velocity_pre,
                               boundary_descriptor_pressure_pre,
                               field_functions_pre,
                               param_pre,
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
                                                       periodic_faces,
                                                       boundary_descriptor_velocity,
                                                       boundary_descriptor_pressure,
                                                       field_functions,
                                                       param,
                                                       mpi_comm));

    navier_stokes_operator = navier_stokes_operator_coupled;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting.reset(new DGDualSplitting(*triangulation,
                                                                    mesh->get_mapping(),
                                                                    periodic_faces,
                                                                    boundary_descriptor_velocity,
                                                                    boundary_descriptor_pressure,
                                                                    field_functions,
                                                                    param,
                                                                    mpi_comm));

    navier_stokes_operator = navier_stokes_operator_dual_splitting;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction.reset(
      new DGPressureCorrection(*triangulation,
                               mesh->get_mapping(),
                               periodic_faces,
                               boundary_descriptor_velocity,
                               boundary_descriptor_pressure,
                               field_functions,
                               param,
                               mpi_comm));

    navier_stokes_operator = navier_stokes_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free 1
  matrix_free_wrapper_pre.reset(new MatrixFreeWrapper<dim, Number>(mesh_pre->get_mapping()));
  matrix_free_wrapper_pre->append_data_structures(*navier_stokes_operator_pre);
  matrix_free_wrapper_pre->reinit(param_pre.use_cell_based_face_loops, triangulation_pre);

  // initialize matrix_free 2
  matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(mesh->get_mapping()));
  matrix_free_wrapper->append_data_structures(*navier_stokes_operator);
  matrix_free_wrapper->reinit(param.use_cell_based_face_loops, triangulation);


  // setup Navier-Stokes operator
  navier_stokes_operator_pre->setup(matrix_free_wrapper_pre);
  navier_stokes_operator->setup(matrix_free_wrapper);

  // setup postprocessor
  postprocessor_pre = application->construct_postprocessor_precursor(param_pre, mpi_comm);
  postprocessor_pre->setup(*navier_stokes_operator_pre);

  postprocessor = application->construct_postprocessor(param, mpi_comm);
  postprocessor->setup(*navier_stokes_operator);


  // Setup time integrator

  if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_pre.reset(new TimeIntCoupled(
      navier_stokes_operator_coupled_pre, param_pre, mpi_comm, postprocessor_pre));
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_pre.reset(new TimeIntDualSplitting(
      navier_stokes_operator_dual_splitting_pre, param_pre, mpi_comm, postprocessor_pre));
  }
  else if(this->param_pre.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pre.reset(new TimeIntPressureCorrection(
      navier_stokes_operator_pressure_correction_pre, param_pre, mpi_comm, postprocessor_pre));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator.reset(
      new TimeIntCoupled(navier_stokes_operator_coupled, param, mpi_comm, postprocessor));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator.reset(new TimeIntDualSplitting(
      navier_stokes_operator_dual_splitting, param, mpi_comm, postprocessor));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator.reset(new TimeIntPressureCorrection(
      navier_stokes_operator_pressure_correction, param, mpi_comm, postprocessor));
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

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::solve() const
{
  // run time loop

  set_start_time();

  synchronize_time_step_size();

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

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::analyze_iterations(
  InputParameters const &        param,
  std::shared_ptr<TimeInt> const time_integrator) const
{
  // Iterations
  if(param.solver_type == SolverType::Unsteady)
  {
    std::vector<std::string> names;
    std::vector<double>      iterations;

    time_integrator->get_iterations(names, iterations);

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }
}
template<int dim, typename Number>
double
DriverPrecursor<dim, Number>::analyze_computing_times(
  InputParameters const &        param,
  std::shared_ptr<TimeInt> const time_integrator) const
{
  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(param.solver_type == SolverType::Unsteady)
  {
    time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter"));
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
    this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  return sum_of_substeps;
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << std::endl
              << "Average number of iterations for incompressible Navier-Stokes solver:"
              << std::endl;

  this->pcout << std::endl << "Precursor domain:" << std::endl;

  analyze_iterations(param_pre, time_integrator_pre);

  this->pcout << std::endl << "Actual domain:" << std::endl;

  analyze_iterations(param, time_integrator);

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  this->pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  // wall times
  this->pcout << std::endl << "Domain 1:" << std::endl;

  double const time_domain_pre_avg = analyze_computing_times(param_pre, time_integrator_pre);

  // wall times
  this->pcout << std::endl << "Domain 2:" << std::endl;

  double const time_domain_avg = analyze_computing_times(param, time_integrator);

  this->pcout << std::endl;

  Utilities::MPI::MinMaxAvg setup_time_data = Utilities::MPI::min_max_avg(setup_time, mpi_comm);
  double const              setup_time_avg  = setup_time_data.avg;
  this->pcout << "  " << std::setw(length) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - time_domain_pre_avg - time_domain_avg - setup_time_avg;
  this->pcout << "  " << std::setw(length) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  this->pcout << std::endl
              << "Computational costs (both domains, including setup + postprocessing):"
              << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs =
    navier_stokes_operator_pre->get_number_of_dofs() + navier_stokes_operator->get_number_of_dofs();

  if(param_pre.solver_type == SolverType::Unsteady)
  {
    unsigned int N_time_steps      = time_integrator_pre->get_number_of_time_steps();
    double const time_per_timestep = overall_time_avg / (double)N_time_steps;
    this->pcout << std::endl
                << "Throughput per time step (both domains, including setup + postprocessing):"
                << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Time steps              = " << std::left << N_time_steps << std::endl
                << "  Wall time per time step = " << std::scientific << std::setprecision(2)
                << time_per_timestep << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter"));
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
