/*
 * driver.cpp
 *
 *  Created on: 31.03.2020
 *      Author: fehn
 */

#include <exadg/incompressible_flow_with_transport/driver.h>
#include <exadg/utilities/print_throughput.h>

namespace ExaDG
{
namespace FTI
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm, unsigned int const n_scalars)
  : mpi_comm(comm),
    n_scalars(n_scalars),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    use_adaptive_time_stepping(false)
{
  scalar_param.resize(n_scalars);
  scalar_field_functions.resize(n_scalars);
  scalar_boundary_descriptor.resize(n_scalars);

  conv_diff_operator.resize(n_scalars);
  scalar_postprocessor.resize(n_scalars);
  scalar_time_integrator.resize(n_scalars);
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                             with scalar transport.                              " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space)
{
  Timer timer;
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  // parameters fluid
  application->set_input_parameters(fluid_param);
  fluid_param.check_input_parameters(pcout);
  AssertThrow(fluid_param.problem_type == IncNS::ProblemType::Unsteady,
              ExcMessage("ProblemType must be unsteady!"));

  fluid_param.print(pcout, "List of input parameters for fluid solver:");

  // parameters scalar
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    application->set_input_parameters_scalar(scalar_param[i], i);
    scalar_param[i].check_input_parameters();
    AssertThrow(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady,
                ExcMessage("ProblemType must be unsteady!"));

    scalar_param[i].print(pcout,
                          "List of input parameters for scalar quantity " +
                            Utilities::to_string(i) + ":");
  }

  // triangulation
  if(fluid_param.triangulation_type == TriangulationType::Distributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].triangulation_type == TriangulationType::Distributed,
                  ExcMessage(
                    "Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(fluid_param.triangulation_type == TriangulationType::FullyDistributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].triangulation_type == TriangulationType::FullyDistributed,
                  ExcMessage(
                    "Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  application->create_grid(triangulation, refine_space, periodic_faces);
  print_grid_data(pcout, refine_space, *triangulation);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(fluid_param.mapping, degree);

  if(fluid_param.ale_formulation) // moving mesh
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].ale_formulation == true,
                  ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));
    }

    AssertThrow(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Analytical,
                ExcMessage("not implemented."));

    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion = application->set_mesh_movement_function();
    moving_mesh.reset(new MovingMeshFunction<dim, Number>(
      *triangulation, mapping_degree, degree, mpi_comm, mesh_motion, fluid_param.start_time));

    mesh = moving_mesh;
  }
  else // static mesh
  {
    mesh.reset(new Mesh<dim>(mapping_degree));
  }

  // boundary conditions
  fluid_boundary_descriptor_velocity.reset(new IncNS::BoundaryDescriptorU<dim>());
  fluid_boundary_descriptor_pressure.reset(new IncNS::BoundaryDescriptorP<dim>());

  application->set_boundary_conditions(fluid_boundary_descriptor_velocity,
                                       fluid_boundary_descriptor_pressure);
  verify_boundary_conditions(*fluid_boundary_descriptor_velocity, *triangulation, periodic_faces);
  verify_boundary_conditions(*fluid_boundary_descriptor_pressure, *triangulation, periodic_faces);

  // field functions
  fluid_field_functions.reset(new IncNS::FieldFunctions<dim>());
  application->set_field_functions(fluid_field_functions);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // boundary conditions
    scalar_boundary_descriptor[i].reset(new ConvDiff::BoundaryDescriptor<dim>());

    application->set_boundary_conditions_scalar(scalar_boundary_descriptor[i], i);
    verify_boundary_conditions(*scalar_boundary_descriptor[i], *triangulation, periodic_faces);

    // field functions
    scalar_field_functions[i].reset(new ConvDiff::FieldFunctions<dim>());
    application->set_field_functions_scalar(scalar_field_functions[i], i);
  }


  // initialize navier_stokes_operator
  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled.reset(new DGCoupled(*triangulation,
                                                       mesh->get_mapping(),
                                                       degree,
                                                       periodic_faces,
                                                       fluid_boundary_descriptor_velocity,
                                                       fluid_boundary_descriptor_pressure,
                                                       fluid_field_functions,
                                                       fluid_param,
                                                       "fluid",
                                                       mpi_comm));

    navier_stokes_operator = navier_stokes_operator_coupled;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting.reset(
      new DGDualSplitting(*triangulation,
                          mesh->get_mapping(),
                          degree,
                          periodic_faces,
                          fluid_boundary_descriptor_velocity,
                          fluid_boundary_descriptor_pressure,
                          fluid_field_functions,
                          fluid_param,
                          "fluid",
                          mpi_comm));

    navier_stokes_operator = navier_stokes_operator_dual_splitting;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction.reset(
      new DGPressureCorrection(*triangulation,
                               mesh->get_mapping(),
                               degree,
                               periodic_faces,
                               fluid_boundary_descriptor_velocity,
                               fluid_boundary_descriptor_pressure,
                               fluid_field_functions,
                               fluid_param,
                               "fluid",
                               mpi_comm));

    navier_stokes_operator = navier_stokes_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    conv_diff_operator[i].reset(new ConvDiff::DGOperator<dim, Number>(*triangulation,
                                                                      mesh->get_mapping(),
                                                                      degree,
                                                                      periodic_faces,
                                                                      scalar_boundary_descriptor[i],
                                                                      scalar_field_functions[i],
                                                                      scalar_param[i],
                                                                      "scalar" + std::to_string(i),
                                                                      mpi_comm));
  }

  // initialize matrix_free
  matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  if(fluid_param.use_cell_based_face_loops)
  {
    auto tria =
      std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation);
    Categorization::do_cell_based_loops(*tria, matrix_free_data->data);
  }
  navier_stokes_operator->fill_matrix_free_data(*matrix_free_data);
  for(unsigned int i = 0; i < n_scalars; ++i)
    conv_diff_operator[i]->fill_matrix_free_data(*matrix_free_data);

  matrix_free.reset(new MatrixFree<dim, Number>());
  matrix_free->reinit(mesh->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(
      scalar_param[i].use_cell_based_face_loops == fluid_param.use_cell_based_face_loops,
      ExcMessage(
        "Parameter use_cell_based_face_loops should be the same for fluid and scalar transport."));
  }

  // setup Navier-Stokes operator
  if(fluid_param.boussinesq_term)
  {
    // assume that the first scalar field with index 0 is the active scalar that
    // couples to the incompressible Navier-Stokes equations
    navier_stokes_operator->setup(matrix_free,
                                  matrix_free_data,
                                  conv_diff_operator[0]->get_dof_name());
  }
  else
  {
    navier_stokes_operator->setup(matrix_free, matrix_free_data);
  }

  // setup convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    conv_diff_operator[i]->setup(matrix_free,
                                 matrix_free_data,
                                 navier_stokes_operator->get_dof_name_velocity());
  }

  // setup postprocessor
  fluid_postprocessor = application->construct_postprocessor(degree, mpi_comm);
  fluid_postprocessor->setup(*navier_stokes_operator);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_postprocessor[i] = application->construct_postprocessor_scalar(degree, mpi_comm, i);
    scalar_postprocessor[i]->setup(conv_diff_operator[i]->get_dof_handler(), mesh->get_mapping());
  }

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_time_integrator.reset(new TimeIntCoupled(navier_stokes_operator_coupled,
                                                   fluid_param,
                                                   0 /* refine_time */,
                                                   mpi_comm,
                                                   fluid_postprocessor,
                                                   moving_mesh,
                                                   matrix_free));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting,
                                                         fluid_param,
                                                         0 /* refine_time */,
                                                         mpi_comm,
                                                         fluid_postprocessor,
                                                         moving_mesh,
                                                         matrix_free));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_time_integrator.reset(
      new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction,
                                    fluid_param,
                                    0 /* refine_time */,
                                    mpi_comm,
                                    fluid_postprocessor,
                                    moving_mesh,
                                    matrix_free));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  fluid_time_integrator->setup(fluid_param.restarted_simulation);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // initialize time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      scalar_time_integrator[i].reset(new ConvDiff::TimeIntExplRK<Number>(conv_diff_operator[i],
                                                                          scalar_param[i],
                                                                          0 /* refine_time */,
                                                                          mpi_comm,
                                                                          scalar_postprocessor[i]));
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      scalar_time_integrator[i].reset(new ConvDiff::TimeIntBDF<dim, Number>(conv_diff_operator[i],
                                                                            scalar_param[i],
                                                                            0 /* refine_time */,
                                                                            mpi_comm,
                                                                            scalar_postprocessor[i],
                                                                            moving_mesh,
                                                                            matrix_free));
    }
    else
    {
      AssertThrow(scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::ExplRK ||
                    scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }

    if(scalar_param[i].restarted_simulation == false)
    {
      // The parameter start_with_low_order has to be true.
      // This is due to the fact that the setup function of the time integrator initializes
      // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
      // start_with_low_order == false. However, the combined time step size
      // is not known at this point since we have to first communicate the time step size
      // in order to find the minimum time step size. Overwriting the time step size would
      // imply that the time step sizes are non constant for a time integrator, but the
      // time integration constants are initialized based on the assumption of constant
      // time step size. Hence, the easiest way to avoid these kind of
      // inconsistencies is to preclude the case start_with_low_order == false.
      // In case of a restart, start_with_low_order = false is possible since it has been
      // enforced that all previous time steps are identical for fluid and scalar transport.
      AssertThrow(fluid_param.start_with_low_order == true &&
                    scalar_param[i].start_with_low_order == true,
                  ExcMessage("start_with_low_order has to be true for this solver."));
    }

    scalar_time_integrator[i]->setup(scalar_param[i].restarted_simulation);

    // adaptive time stepping
    if(fluid_param.adaptive_time_stepping == true)
    {
      AssertThrow(
        scalar_param[i].adaptive_time_stepping == true,
        ExcMessage(
          "Adaptive time stepping has to be used for both fluid and scalar transport solvers."));

      use_adaptive_time_stepping = true;
    }
  }

  // setup solvers once time integrator has been initialized
  navier_stokes_operator->setup_solvers(
    fluid_time_integrator->get_scaling_factor_time_derivative_term(),
    fluid_time_integrator->get_velocity());

  // setup solvers in case of BDF time integration (solution of linear systems of equations)
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(scalar_param[i].analytical_velocity_field == false,
                ExcMessage(
                  "An analytical velocity field can not be used for this coupled solver."));

    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> scalar_time_integrator_BDF =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      double const scaling_factor =
        scalar_time_integrator_BDF->get_scaling_factor_time_derivative_term();

      LinearAlgebra::distributed::Vector<Number> vector;
      navier_stokes_operator->initialize_vector_velocity(vector);
      LinearAlgebra::distributed::Vector<Number> const * velocity = &vector;

      conv_diff_operator[i]->setup_solver(scaling_factor, velocity);
    }
    else
    {
      AssertThrow(scalar_param[i].temporal_discretization ==
                    ConvDiff::TemporalDiscretization::ExplRK,
                  ExcMessage("Not implemented."));
    }
  }

  // Boussinesq term
  // assume that the first scalar quantity with index 0 is the active scalar coupled to
  // the incompressible Navier-Stokes equations via the Boussinesq term
  if(fluid_param.boussinesq_term)
    conv_diff_operator[0]->initialize_dof_vector(temperature);

  timer_tree.insert({"Flow + transport", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const fluid_time = fluid_time_integrator->get_time();

  double time = fluid_time;

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double const scalar_time = scalar_time_integrator[i]->get_time();
    time                     = std::min(time, scalar_time);
  }

  // Set the same start time for both solvers

  // fluid
  fluid_time_integrator->reset_time(time);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->reset_time(time);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_fluid = std::numeric_limits<double>::max();

  // fluid
  if(fluid_time_integrator->get_time() > fluid_param.start_time - EPSILON)
    time_step_size_fluid = fluid_time_integrator->get_time_step_size();

  double time_step_size = time_step_size_fluid;

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double time_step_size_scalar = std::numeric_limits<double>::max();
    if(scalar_time_integrator[i]->get_time() > scalar_param[i].start_time - EPSILON)
      time_step_size_scalar = scalar_time_integrator[i]->get_time_step_size();

    time_step_size = std::min(time_step_size, time_step_size_scalar);
  }

  if(use_adaptive_time_stepping == false)
  {
    // decrease time_step in order to exactly hit end_time
    time_step_size = (fluid_param.end_time - fluid_param.start_time) /
                     (1 + int((fluid_param.end_time - fluid_param.start_time) / time_step_size));

    pcout << std::endl << "Combined time step size dt = " << time_step_size << std::endl;
  }

  // Set the same time step size for both solvers

  // fluid
  fluid_time_integrator->set_current_time_step_size(time_step_size);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->set_current_time_step_size(time_step_size);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_scalar_to_fluid() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // scalar transport solver (scalar 0 by definition) for the temperature and hand it over to the
  // fluid solver.
  if(fluid_param.boussinesq_term)
  {
    // assume that the first scalar quantity with index 0 is the active scalar coupled to
    // the incompressible Navier-Stokes equations via the Boussinesq term
    if(scalar_param[0].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else if(scalar_param[0].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    navier_stokes_operator->set_temperature(temperature);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_fluid_to_all_scalars() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // fluid solver for the velocity field and hand it over to all scalar transport solvers.
  std::vector<LinearAlgebra::distributed::Vector<Number> const *> velocities;
  std::vector<double>                                             times;
  fluid_time_integrator->get_velocities_and_times_np(velocities, times);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  Timer timer;
  timer.restart();

  Timer sub_timer;

  sub_timer.restart();
  moving_mesh->move_mesh(fluid_time_integrator->get_next_time());
  timer_tree.insert({"Flow + transport", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  matrix_free->update_mapping(moving_mesh->get_mapping());
  timer_tree.insert({"Flow + transport", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  navier_stokes_operator->update_after_mesh_movement();
  for(unsigned int i = 0; i < n_scalars; ++i)
    conv_diff_operator[i]->update_after_mesh_movement();
  timer_tree.insert({"Flow + transport", "ALE", "Update all operators"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_time_integrator->ale_update();
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_bdf =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
    time_int_bdf->ale_update();
  }
  timer_tree.insert({"Flow + transport", "ALE", "Update all time integrators"},
                    sub_timer.wall_time());

  timer_tree.insert({"Flow + transport", "ALE"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // time loop
  bool finished = false;
  while(not finished)
  {
    /*
     * pre solve
     */
    fluid_time_integrator->advance_one_timestep_pre_solve(true);
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_pre_solve(false);

    /*
     * ALE: move the mesh and update dependent data structures
     */
    if(fluid_param.ale_formulation) // moving mesh
      ale_update();

    /*
     *  solve
     */

    // Communicate scalar -> fluid
    communicate_scalar_to_fluid();

    // fluid: advance one time step
    fluid_time_integrator->advance_one_timestep_solve();

    // Communicate fluid -> all scalars
    communicate_fluid_to_all_scalars();

    // scalar transport: advance one time step
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_solve();

    /*
     * post solve
     */
    fluid_time_integrator->advance_one_timestep_post_solve();
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_post_solve();

    // Both solvers have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();

    finished = fluid_time_integrator->finished();
    for(unsigned int i = 0; i < n_scalars; ++i)
      finished = finished && scalar_time_integrator[i]->finished();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_statistics(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // Iterations
  this->pcout << std::endl << "Average number of iterations:" << std::endl;

  // Fluid
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  this->fluid_time_integrator->print_iterations();

  // Scalar
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // only relevant for BDF time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_integrator_bdf->print_iterations();
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      this->pcout << "  Explicit solver (no systems of equations have to be solved)" << std::endl;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // Wall times

  timer_tree.insert({"Flow + transport"}, total_time);

  timer_tree.insert({"Flow + transport"}, fluid_time_integrator->get_timings(), "Timeloop fluid");

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    timer_tree.insert({"Flow + transport"},
                      scalar_time_integrator[i]->get_timings(),
                      "Timeloop scalar " + std::to_string(i));
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index DoFs = this->navier_stokes_operator->get_number_of_dofs();

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    DoFs += this->conv_diff_operator[i]->get_number_of_dofs();
  }

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  unsigned int const N_time_steps = fluid_time_integrator->get_number_of_time_steps();
  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FTI
} // namespace ExaDG
