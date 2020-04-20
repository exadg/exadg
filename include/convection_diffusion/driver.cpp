/*
 * driver.cpp
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#include "driver.h"

namespace ConvDiff
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    overall_time(0.0),
    setup_time(0.0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                          convection-diffusion equation                          " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space,
                           unsigned int const &                          refine_time)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);
  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  // triangulation
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

  application->create_grid(triangulation, refine_space, periodic_faces);
  print_grid_data(pcout, refine_space, *triangulation);

  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  application->set_boundary_conditions(boundary_descriptor);
  verify_boundary_conditions(*boundary_descriptor, *triangulation, periodic_faces);

  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);

  if(param.ale_formulation) // moving mesh
  {
    std::shared_ptr<Function<dim>> mesh_motion = application->set_mesh_movement_function();
    moving_mesh.reset(new MovingMeshAnalytical<dim, Number>(
      *triangulation, mapping_degree, degree, mpi_comm, mesh_motion, param.start_time));

    mesh = moving_mesh;
  }
  else // static mesh
  {
    mesh.reset(new Mesh<dim>(mapping_degree));
  }

  // initialize convection-diffusion operator
  conv_diff_operator.reset(new DGOperator<dim, Number>(*triangulation,
                                                       mesh->get_mapping(),
                                                       degree,
                                                       periodic_faces,
                                                       boundary_descriptor,
                                                       field_functions,
                                                       param,
                                                       mpi_comm));

  // initialize matrix_free
  matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(mesh->get_mapping()));
  matrix_free_wrapper->append_data_structures(*conv_diff_operator);
  matrix_free_wrapper->reinit(param.use_cell_based_face_loops, triangulation);

  // setup convection-diffusion operator
  conv_diff_operator->setup(matrix_free_wrapper);

  // initialize postprocessor
  postprocessor = application->construct_postprocessor(degree, mpi_comm);
  postprocessor->setup(conv_diff_operator->get_dof_handler(), mesh->get_mapping());

  // initialize time integrator or driver for steady problems
  if(param.problem_type == ProblemType::Unsteady)
  {
    if(param.temporal_discretization == TemporalDiscretization::ExplRK)
    {
      time_integrator.reset(
        new TimeIntExplRK<Number>(conv_diff_operator, param, refine_time, mpi_comm, postprocessor));
    }
    else if(param.temporal_discretization == TemporalDiscretization::BDF)
    {
      time_integrator.reset(new TimeIntBDF<dim, Number>(conv_diff_operator,
                                                        param,
                                                        refine_time,
                                                        mpi_comm,
                                                        postprocessor,
                                                        moving_mesh,
                                                        matrix_free_wrapper));
    }
    else
    {
      AssertThrow(param.temporal_discretization == TemporalDiscretization::ExplRK ||
                    param.temporal_discretization == TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }

    time_integrator->setup(param.restarted_simulation);
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady.reset(
      new DriverSteadyProblems<Number>(conv_diff_operator, param, mpi_comm, postprocessor));
    driver_steady->setup();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  // setup solvers in case of BDF time integration or steady problems
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  VectorType const *                                 velocity_ptr = nullptr;
  VectorType                                         velocity;

  if(param.problem_type == ProblemType::Unsteady)
  {
    if(param.temporal_discretization == TemporalDiscretization::BDF)
    {
      std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);

      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        conv_diff_operator->initialize_dof_vector_velocity(velocity);
        conv_diff_operator->interpolate_velocity(velocity, time_integrator->get_time());
        velocity_ptr = &velocity;
      }

      conv_diff_operator->setup_solver(
        time_integrator_bdf->get_scaling_factor_time_derivative_term(), velocity_ptr);
    }
    else
    {
      AssertThrow(param.temporal_discretization == TemporalDiscretization::ExplRK,
                  ExcMessage("Not implemented."));
    }
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      conv_diff_operator->initialize_dof_vector_velocity(velocity);
      conv_diff_operator->interpolate_velocity(velocity, 0.0 /* time */);
      velocity_ptr = &velocity;
    }

    conv_diff_operator->setup_solver(1.0 /* scaling_factor_time_derivative_term */, velocity_ptr);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  if(param.problem_type == ProblemType::Unsteady)
  {
    if(this->param.ale_formulation == true)
    {
      do
      {
        time_integrator->advance_one_timestep_pre_solve();

        // move the mesh and update dependent data structures
        std::shared_ptr<TimeIntBDF<dim, Number>> time_int_bdf =
          std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);
        moving_mesh->move_mesh(time_int_bdf->get_next_time());
        matrix_free_wrapper->update_mapping();
        conv_diff_operator->update_after_mesh_movement();
        time_int_bdf->ale_update();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      } while(!time_integrator->finished());
    }
    else
    {
      time_integrator->timeloop();
    }
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->solve_problem();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  Operatortype operator_type;
  string_to_enum(operator_type, operator_type_string);

  LinearAlgebra::distributed::Vector<Number> dst, src;

  conv_diff_operator->initialize_dof_vector(src);
  src = 1.0;
  conv_diff_operator->initialize_dof_vector(dst);

  LinearAlgebra::distributed::Vector<Number> velocity;
  if(param.convective_problem())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      conv_diff_operator->initialize_dof_vector_velocity(velocity);
      velocity = 1.0;
    }
  }

  if(operator_type == Operatortype::ConvectiveOperator)
    conv_diff_operator->update_convective_term(1.0 /* time */, &velocity);
  else if(operator_type == Operatortype::MassConvectionDiffusionOperator)
    conv_diff_operator->update_conv_diff_operator(1.0 /* time */,
                                                  1.0 /* scaling_factor_mass_matrix */,
                                                  &velocity);

  // Timer and wall times
  Timer  timer;
  double wall_time = std::numeric_limits<double>::max();

  for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
  {
    double current_wall_time = 0.0;

    // apply matrix-vector product several times
    for(unsigned int i = 0; i < n_repetitions_inner; ++i)
    {
      timer.restart();

      if(operator_type == Operatortype::MassOperator)
        conv_diff_operator->apply_mass_matrix(dst, src);
      else if(operator_type == Operatortype::ConvectiveOperator)
        conv_diff_operator->apply_convective_term(dst, src);
      else if(operator_type == Operatortype::DiffusiveOperator)
        conv_diff_operator->apply_diffusive_term(dst, src);
      else if(operator_type == Operatortype::MassConvectionDiffusionOperator)
        conv_diff_operator->apply_conv_diff_operator(dst, src);

      current_wall_time += timer.wall_time();
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time, current_wall_time);
  }

  if(wall_time * n_repetitions_inner * n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout
      << std::endl
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducible results."
      << std::endl;
  }

  types::global_dof_index dofs              = conv_diff_operator->get_number_of_dofs();
  double                  dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(
    conv_diff_operator->get_polynomial_degree(), dofs, dofs_per_walltime);
}

template<int dim, typename Number>
void
Driver<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for convection-diffusion solver:" << std::endl;

  // Iterations are only relevant for BDF time integrator
  if(param.temporal_discretization == TemporalDiscretization::BDF)
  {
    // Iterations
    if(param.problem_type == ProblemType::Unsteady)
    {
      this->pcout << std::endl << "Average number of iterations:" << std::endl;

      std::vector<std::string> names;
      std::vector<double>      iterations;

      std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<TimeIntBDF<dim, Number>>(time_integrator);
      time_integrator_bdf->get_iterations(names, iterations);

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
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(param.problem_type == ProblemType::Unsteady)
  {
    this->time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    this->driver_steady->get_wall_times(names, computing_times);
  }

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

  double const other = overall_time_avg - sum_of_substeps - setup_time_avg;
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
  types::global_dof_index const DoFs = conv_diff_operator->get_number_of_dofs();

  if(param.problem_type == ProblemType::Unsteady)
  {
    unsigned int N_time_steps      = this->time_integrator->get_number_of_time_steps();
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
  }
  else
  {
    this->pcout << std::endl
                << "Throughput (including setup + postprocessing):" << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (overall_time_avg * N_mpi_processes) << " DoFs/s/core" << std::endl;
  }


  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace ConvDiff
