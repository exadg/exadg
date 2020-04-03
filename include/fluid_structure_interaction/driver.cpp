/*
 * driver.cpp
 *
 *  Created on: 01.04.2020
 *      Author: fehn
 */

#include "driver.h"

namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
    overall_time(0.0),
    setup_time(0.0),
    ale_update_time(0.0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "           High-order solver for fluid-structure interaction problems            " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree_fluid,
                           unsigned int const &                          degree_poisson,
                           unsigned int const &                          refine_space)

{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  // parameters fluid
  application->set_input_parameters_fluid(fluid_param);
  fluid_param.check_input_parameters(pcout);
  fluid_param.print(pcout, "List of input parameters for incompressible flow solver:");

  // parameters Poisson
  application->set_input_parameters_poisson(poisson_param);
  poisson_param.check_input_parameters();
  poisson_param.print(pcout, "List of input parameters for Poisson solver (moving mesh):");

  // Some FSI specific Asserts
  AssertThrow(fluid_param.problem_type == IncNS::ProblemType::Unsteady,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));
  AssertThrow(fluid_param.ale_formulation == true,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // triangulation
  if(fluid_param.triangulation_type == TriangulationType::Distributed)
  {
    fluid_triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(fluid_param.triangulation_type == TriangulationType::FullyDistributed)
  {
    fluid_triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  application->create_grid_fluid(fluid_triangulation, refine_space, fluid_periodic_faces);
  print_grid_data(pcout, refine_space, *fluid_triangulation);

  // field functions and boundary conditions

  // fluid
  fluid_boundary_descriptor_velocity.reset(new IncNS::BoundaryDescriptorU<dim>());
  fluid_boundary_descriptor_pressure.reset(new IncNS::BoundaryDescriptorP<dim>());
  application->set_boundary_conditions_fluid(fluid_boundary_descriptor_velocity,
                                             fluid_boundary_descriptor_pressure);
  verify_boundary_conditions(*fluid_boundary_descriptor_velocity,
                             *fluid_triangulation,
                             fluid_periodic_faces);
  verify_boundary_conditions(*fluid_boundary_descriptor_pressure,
                             *fluid_triangulation,
                             fluid_periodic_faces);

  fluid_field_functions.reset(new IncNS::FieldFunctions<dim>());
  application->set_field_functions_fluid(fluid_field_functions);

  // poisson
  poisson_boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<1, dim>());
  application->set_boundary_conditions_poisson(poisson_boundary_descriptor);
  verify_boundary_conditions(*poisson_boundary_descriptor,
                             *fluid_triangulation,
                             fluid_periodic_faces);

  poisson_field_functions.reset(new Poisson::FieldFunctions<dim>());
  application->set_field_functions_poisson(poisson_field_functions);

  AssertThrow(poisson_param.right_hand_side == false,
              ExcMessage("Parameter does not make sense in context of FSI."));

  // mapping for Poisson solver (static mesh)
  unsigned int const mapping_degree = get_mapping_degree(poisson_param.mapping, degree_poisson);
  poisson_mesh.reset(new Mesh<dim>(mapping_degree));

  // initialize Poisson operator
  poisson_operator.reset(new Poisson::Operator<dim, Number, dim>(*fluid_triangulation,
                                                                 poisson_mesh->get_mapping(),
                                                                 degree_poisson,
                                                                 fluid_periodic_faces,
                                                                 poisson_boundary_descriptor,
                                                                 poisson_field_functions,
                                                                 poisson_param,
                                                                 mpi_comm));

  // initialize matrix_free
  poisson_matrix_free_wrapper.reset(
    new MatrixFreeWrapper<dim, Number>(poisson_mesh->get_mapping()));
  poisson_matrix_free_wrapper->append_data_structures(*poisson_operator);
  poisson_matrix_free_wrapper->reinit(poisson_param.enable_cell_based_face_loops,
                                      fluid_triangulation);

  poisson_operator->setup(poisson_matrix_free_wrapper);
  poisson_operator->setup_solver();

  // mapping for fluid problem (moving mesh)
  {
    unsigned int const mapping_degree = get_mapping_degree(fluid_param.mapping, degree_fluid);

    // TODO
    if(fluid_param.ale_formulation) // moving mesh
    {
      fluid_moving_mesh.reset(new MovingMeshPoisson<dim, Number>(
        mapping_degree, mpi_comm, poisson_operator, fluid_param.start_time));

      //      std::shared_ptr<Function<dim>> mesh_motion =
      //      application->set_mesh_movement_function_fluid(); fluid_moving_mesh.reset(new
      //      MovingMeshAnalytical<dim, Number>(
      //        *fluid_triangulation, mapping_degree, fluid_param.degree_u, mpi_comm, mesh_motion,
      //        fluid_param.start_time));

      fluid_mesh = fluid_moving_mesh;
    }
    else // static mesh
    {
      fluid_mesh.reset(new Mesh<dim>(mapping_degree));
    }
  }

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_operator_coupled.reset(new DGCoupled(*fluid_triangulation,
                                               fluid_mesh->get_mapping(),
                                               degree_fluid,
                                               fluid_periodic_faces,
                                               fluid_boundary_descriptor_velocity,
                                               fluid_boundary_descriptor_pressure,
                                               fluid_field_functions,
                                               fluid_param,
                                               mpi_comm));

    fluid_operator = fluid_operator_coupled;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_operator_dual_splitting.reset(new DGDualSplitting(*fluid_triangulation,
                                                            fluid_mesh->get_mapping(),
                                                            degree_fluid,
                                                            fluid_periodic_faces,
                                                            fluid_boundary_descriptor_velocity,
                                                            fluid_boundary_descriptor_pressure,
                                                            fluid_field_functions,
                                                            fluid_param,
                                                            mpi_comm));

    fluid_operator = fluid_operator_dual_splitting;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_operator_pressure_correction.reset(
      new DGPressureCorrection(*fluid_triangulation,
                               fluid_mesh->get_mapping(),
                               degree_fluid,
                               fluid_periodic_faces,
                               fluid_boundary_descriptor_velocity,
                               fluid_boundary_descriptor_pressure,
                               fluid_field_functions,
                               fluid_param,
                               mpi_comm));

    fluid_operator = fluid_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  fluid_matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(fluid_mesh->get_mapping()));
  fluid_matrix_free_wrapper->append_data_structures(*fluid_operator);
  fluid_matrix_free_wrapper->reinit(fluid_param.use_cell_based_face_loops, fluid_triangulation);

  // setup Navier-Stokes operator
  fluid_operator->setup(fluid_matrix_free_wrapper);

  // setup postprocessor
  fluid_postprocessor = application->construct_postprocessor_fluid(degree_fluid, mpi_comm);
  fluid_postprocessor->setup(*fluid_operator);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_time_integrator.reset(new TimeIntCoupled(fluid_operator_coupled,
                                                   fluid_param,
                                                   0 /* refine_time */,
                                                   mpi_comm,
                                                   fluid_postprocessor,
                                                   fluid_moving_mesh,
                                                   fluid_matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_time_integrator.reset(new TimeIntDualSplitting(fluid_operator_dual_splitting,
                                                         fluid_param,
                                                         0 /* refine_time */,
                                                         mpi_comm,
                                                         fluid_postprocessor,
                                                         fluid_moving_mesh,
                                                         fluid_matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_time_integrator.reset(new TimeIntPressureCorrection(fluid_operator_pressure_correction,
                                                              fluid_param,
                                                              0 /* refine_time */,
                                                              mpi_comm,
                                                              fluid_postprocessor,
                                                              fluid_moving_mesh,
                                                              fluid_matrix_free_wrapper));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  fluid_time_integrator->setup(fluid_param.restarted_simulation);

  fluid_operator->setup_solvers(fluid_time_integrator->get_scaling_factor_time_derivative_term(),
                                fluid_time_integrator->get_velocity());

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  do
  {
    fluid_time_integrator->advance_one_timestep_pre_solve();

    // partitioned iteration
    unsigned int const N_ITER_MAX = 1;
    for(unsigned int iter = 0; iter < N_ITER_MAX; ++iter)
    {
      // TODO
      LinearAlgebra::distributed::Vector<Number> vec_displacements, vec_velocity;
      if(iter == 0)
      {
        // extrapolate structural displacements and fluid velocity at interface
      }
      else
      {
        // use structural displacements of last iteration and compute
        // fluid velocity at interface using the new structural displacements
      }

      // move the mesh and update dependent data structures
      Timer timer;
      timer.restart();

      // structure_to_moving_mesh->update_data(vec_displacements);
      fluid_moving_mesh->move_mesh(fluid_time_integrator->get_next_time());
      fluid_matrix_free_wrapper->update_mapping();
      fluid_operator->update_after_mesh_movement();
      fluid_time_integrator->ale_update();

      ale_update_time += timer.wall_time();

      // structure_to_fluid->update_data(vec_velocity);
      fluid_time_integrator->advance_one_timestep_solve();

      // TODO update stress boundary condition for solid

      // TODO solve structural problem
    }

    fluid_time_integrator->advance_one_timestep_post_solve();
  } while(!fluid_time_integrator->finished());

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // Iterations
  {
    this->pcout << std::endl << "Average number of iterations:" << std::endl;

    std::vector<std::string> names;
    std::vector<double>      iterations;

    fluid_time_integrator->get_iterations(names, iterations);

    unsigned int length = 1;
    for(unsigned int i = 0; i < names.size(); ++i)
    {
      length = length > names[i].length() ? length : names[i].length();
    }

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  fluid_time_integrator->get_wall_times(names, computing_times);

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data = Utilities::MPI::min_max_avg(setup_time, mpi_comm);
  double const              setup_time_avg  = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  Utilities::MPI::MinMaxAvg ale_time_data = Utilities::MPI::min_max_avg(ale_update_time, mpi_comm);
  double const              ale_time_avg  = ale_time_data.avg;
  if(this->fluid_param.ale_formulation)
  {
    this->pcout << "  " << std::setw(length + 2) << std::left << "ALE update"
                << std::setprecision(2) << std::scientific << std::setw(10) << std::right
                << ale_time_avg << " s  " << std::setprecision(2) << std::fixed << std::setw(6)
                << std::right << ale_time_avg / overall_time_avg * 100 << " %" << std::endl;
  }

  double other = overall_time_avg - sum_of_substeps - setup_time_avg;
  if(this->fluid_param.ale_formulation)
    other -= ale_time_avg;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  this->pcout << std::endl
              << "Computational costs (including setup + postprocessing):" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs = this->fluid_operator->get_number_of_dofs();

  unsigned int N_time_steps      = fluid_time_integrator->get_number_of_time_steps();
  double const time_per_timestep = overall_time_avg / (double)N_time_steps;
  this->pcout << std::endl
              << "Throughput per time step (including setup + postprocessing):" << std::endl
              << "  Degrees of freedom      = " << DoFs << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Time steps              = " << std::left << N_time_steps << std::endl
              << "  Wall time per time step = " << std::scientific << std::setprecision(2)
              << time_per_timestep << " s" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;


  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FSI
