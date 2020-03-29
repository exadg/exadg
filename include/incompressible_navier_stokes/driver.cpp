/*
 * driver.cpp
 *
 *  Created on: 27.03.2020
 *      Author: fehn
 */

#include "driver.h"

namespace IncNS
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
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                     incompressible Navier-Stokes equations                      " << std::endl
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
  // some parameters have to be overwritten
  param.degree_u       = degree;
  param.h_refinements  = refine_space;
  param.dt_refinements = refine_time;

  param.check_input_parameters(pcout);
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

  application->create_grid(triangulation, param.h_refinements, periodic_faces);
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);
  verify_boundary_conditions(*boundary_descriptor_velocity, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_pressure, *triangulation, periodic_faces);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, param.degree_u);

  if(param.ale_formulation) // moving mesh
  {
    std::shared_ptr<Function<dim>> mesh_motion = application->set_mesh_movement_function();
    moving_mesh.reset(new MovingMeshAnalytical<dim, Number>(
      *triangulation, mapping_degree, param.degree_u, mpi_comm, mesh_motion, param.start_time));

    mesh = moving_mesh;
  }
  else // static mesh
  {
    mesh.reset(new Mesh<dim>(mapping_degree));
  }

  if(param.solver_type == SolverType::Unsteady)
  {
    // initialize navier_stokes_operator
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
  }
  else if(param.solver_type == SolverType::Steady)
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
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(mesh->get_mapping()));
  matrix_free_wrapper->append_data_structures(*navier_stokes_operator);
  matrix_free_wrapper->reinit(param.use_cell_based_face_loops, triangulation);

  // setup Navier-Stokes operator
  navier_stokes_operator->setup(matrix_free_wrapper);

  // setup postprocessor
  postprocessor = application->construct_postprocessor(param, mpi_comm);
  postprocessor->setup(*navier_stokes_operator);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  if(param.solver_type == SolverType::Unsteady)
  {
    // initialize navier_stokes_operator
    if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      time_integrator.reset(new TimeIntCoupled(navier_stokes_operator_coupled,
                                               param,
                                               mpi_comm,
                                               postprocessor,
                                               moving_mesh,
                                               matrix_free_wrapper));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting,
                                                     param,
                                                     mpi_comm,
                                                     postprocessor,
                                                     moving_mesh,
                                                     matrix_free_wrapper));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      time_integrator.reset(
        new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction,
                                      param,
                                      mpi_comm,
                                      postprocessor,
                                      moving_mesh,
                                      matrix_free_wrapper));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
  else if(param.solver_type == SolverType::Steady)
  {
    // initialize driver for steady state problem that depends on navier_stokes_operator
    driver_steady.reset(
      new DriverSteady(navier_stokes_operator_coupled, param, mpi_comm, postprocessor));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(param.solver_type == SolverType::Unsteady)
  {
    time_integrator->setup(param.restarted_simulation);

    navier_stokes_operator->setup_solvers(
      time_integrator->get_scaling_factor_time_derivative_term(), time_integrator->get_velocity());
  }
  else if(param.solver_type == SolverType::Steady)
  {
    driver_steady->setup();

    navier_stokes_operator->setup_solvers(1.0 /* dummy */, driver_steady->get_velocity());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(this->param.problem_type == ProblemType::Unsteady)
  {
    // stability analysis (uncomment if desired)
    // time_integrator->postprocessing_stability_analysis();

    if(this->param.ale_formulation == true)
    {
      do
      {
        time_integrator->advance_one_timestep_pre_solve();

        // move the mesh and update dependent data structures
        Timer timer;
        timer.restart();

        moving_mesh->move_mesh(time_integrator->get_next_time());
        matrix_free_wrapper->update_geometry();
        navier_stokes_operator->update_after_mesh_movement();
        time_integrator->ale_update();

        ale_update_time += timer.wall_time();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      } while(!time_integrator->finished());
    }
    else
    {
      time_integrator->timeloop();
    }
  }
  else if(this->param.problem_type == ProblemType::Steady)
  {
    if(param.solver_type == SolverType::Unsteady)
    {
      time_integrator->timeloop_steady_problem();
    }
    else if(param.solver_type == SolverType::Steady)
    {
      // solve steady problem
      driver_steady->solve_steady_problem();
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

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

  this->pcout << "Performance results for incompressible Navier-Stokes solver:" << std::endl;

  // Iterations
  if(param.solver_type == SolverType::Unsteady)
  {
    this->pcout << std::endl << "Average number of iterations:" << std::endl;

    std::vector<std::string> names;
    std::vector<double>      iterations;

    this->time_integrator->get_iterations(names, iterations);

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

  if(param.solver_type == SolverType::Unsteady)
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

  Utilities::MPI::MinMaxAvg ale_time_data = Utilities::MPI::min_max_avg(ale_update_time, mpi_comm);
  double const              ale_time_avg  = ale_time_data.avg;
  if(this->param.ale_formulation)
  {
    this->pcout << "  " << std::setw(length + 2) << std::left << "ALE update"
                << std::setprecision(2) << std::scientific << std::setw(10) << std::right
                << ale_time_avg << " s  " << std::setprecision(2) << std::fixed << std::setw(6)
                << std::right << ale_time_avg / overall_time_avg * 100 << " %" << std::endl;
  }

  double other = overall_time_avg - sum_of_substeps - setup_time_avg;
  if(this->param.ale_formulation)
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
  types::global_dof_index const DoFs = this->navier_stokes_operator->get_number_of_dofs();

  if(param.solver_type == SolverType::Unsteady)
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

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  Operator operator_type;
  string_to_enum(operator_type, operator_type_string);

  AssertThrow(param.degree_p == DegreePressure::MixedOrder,
              ExcMessage(
                "The function get_dofs_per_element() assumes mixed-order polynomials for "
                "velocity and pressure. Additional operator types have to be introduced to "
                "enable equal-order polynomials for this throughput study."));

  // check that the operator type is consistent with the solution approach (coupled vs. splitting)
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    AssertThrow(operator_type == Operator::ConvectiveOperator ||
                  operator_type == Operator::CoupledNonlinearResidual ||
                  operator_type == Operator::CoupledLinearized ||
                  operator_type == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for coupled solution approach."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    AssertThrow(operator_type == Operator::ConvectiveOperator ||
                  operator_type == Operator::PressurePoissonOperator ||
                  operator_type == Operator::HelmholtzOperator ||
                  operator_type == Operator::ProjectionOperator ||
                  operator_type == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for dual splitting scheme."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    AssertThrow(operator_type == Operator::ConvectiveOperator ||
                  operator_type == Operator::PressurePoissonOperator ||
                  operator_type == Operator::VelocityConvDiffOperator ||
                  operator_type == Operator::ProjectionOperator ||
                  operator_type == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for pressure-correction scheme."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // Vectors needed for coupled solution approach
  LinearAlgebra::distributed::BlockVector<Number> dst1, src1;

  // ... for dual splitting, pressure-correction.
  LinearAlgebra::distributed::Vector<Number> dst2, src2;

  // set velocity required for evaluation of linearized operators
  LinearAlgebra::distributed::Vector<Number> velocity;
  navier_stokes_operator->initialize_vector_velocity(velocity);
  velocity = 1.0;
  navier_stokes_operator->set_velocity_ptr(velocity);

  // initialize vectors
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled->initialize_block_vector_velocity_pressure(dst1);
    navier_stokes_operator_coupled->initialize_block_vector_velocity_pressure(src1);
    src1 = 1.0;

    if(operator_type == Operator::ConvectiveOperator ||
       operator_type == Operator::InverseMassMatrix)
    {
      navier_stokes_operator_coupled->initialize_vector_velocity(src2);
      navier_stokes_operator_coupled->initialize_vector_velocity(dst2);
    }
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(operator_type == Operator::ConvectiveOperator ||
       operator_type == Operator::HelmholtzOperator ||
       operator_type == Operator::ProjectionOperator ||
       operator_type == Operator::InverseMassMatrix)
    {
      navier_stokes_operator_dual_splitting->initialize_vector_velocity(src2);
      navier_stokes_operator_dual_splitting->initialize_vector_velocity(dst2);
    }
    else if(operator_type == Operator::PressurePoissonOperator)
    {
      navier_stokes_operator_dual_splitting->initialize_vector_pressure(src2);
      navier_stokes_operator_dual_splitting->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    if(operator_type == Operator::VelocityConvDiffOperator ||
       operator_type == Operator::ProjectionOperator ||
       operator_type == Operator::InverseMassMatrix)
    {
      navier_stokes_operator_pressure_correction->initialize_vector_velocity(src2);
      navier_stokes_operator_pressure_correction->initialize_vector_velocity(dst2);
    }
    else if(operator_type == Operator::PressurePoissonOperator)
    {
      navier_stokes_operator_pressure_correction->initialize_vector_pressure(src2);
      navier_stokes_operator_pressure_correction->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

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

      // clang-format off
      if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        if(operator_type == Operator::CoupledNonlinearResidual)
          navier_stokes_operator_coupled->evaluate_nonlinear_residual(dst1,src1,&src1.block(0), 0.0, 1.0);
        else if(operator_type == Operator::CoupledLinearized)
          navier_stokes_operator_coupled->apply_linearized_problem(dst1,src1, 0.0, 1.0);
        else if(operator_type == Operator::ConvectiveOperator)
          navier_stokes_operator_coupled->evaluate_convective_term(dst2,src2,0.0);
        else if(operator_type == Operator::InverseMassMatrix)
          navier_stokes_operator_coupled->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        if(operator_type == Operator::HelmholtzOperator)
          navier_stokes_operator_dual_splitting->apply_helmholtz_operator(dst2,src2);
        else if(operator_type == Operator::ConvectiveOperator)
          navier_stokes_operator_dual_splitting->evaluate_convective_term(dst2,src2,0.0);
        else if(operator_type == Operator::ProjectionOperator)
          navier_stokes_operator_dual_splitting->apply_projection_operator(dst2,src2);
        else if(operator_type == Operator::PressurePoissonOperator)
          navier_stokes_operator_dual_splitting->apply_laplace_operator(dst2,src2);
        else if(operator_type == Operator::InverseMassMatrix)
          navier_stokes_operator_dual_splitting->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
      {
        if(operator_type == Operator::VelocityConvDiffOperator)
          navier_stokes_operator_pressure_correction->apply_momentum_operator(dst2,src2);
        else if(operator_type == Operator::ProjectionOperator)
          navier_stokes_operator_pressure_correction->apply_projection_operator(dst2,src2);
        else if(operator_type == Operator::PressurePoissonOperator)
          navier_stokes_operator_pressure_correction->apply_laplace_operator(dst2,src2);
        else if(operator_type == Operator::InverseMassMatrix)
          navier_stokes_operator_pressure_correction->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else
      {
        AssertThrow(false,ExcMessage("Not implemented."));
      }
      // clang-format on

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

  types::global_dof_index dofs      = 0;
  unsigned int            fe_degree = 1;

  if(operator_type == Operator::CoupledNonlinearResidual ||
     operator_type == Operator::CoupledLinearized)
  {
    dofs = navier_stokes_operator->get_dof_handler_u().n_dofs() +
           navier_stokes_operator->get_dof_handler_p().n_dofs();

    fe_degree = param.degree_u;
  }
  else if(operator_type == Operator::ConvectiveOperator ||
          operator_type == Operator::VelocityConvDiffOperator ||
          operator_type == Operator::HelmholtzOperator ||
          operator_type == Operator::ProjectionOperator ||
          operator_type == Operator::InverseMassMatrix)
  {
    dofs = navier_stokes_operator->get_dof_handler_u().n_dofs();

    fe_degree = param.degree_u;
  }
  else if(operator_type == Operator::PressurePoissonOperator)
  {
    dofs = navier_stokes_operator->get_dof_handler_p().n_dofs();

    fe_degree = param.get_degree_p();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(fe_degree,
                                                                   dofs,
                                                                   dofs_per_walltime);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;
} // namespace IncNS
