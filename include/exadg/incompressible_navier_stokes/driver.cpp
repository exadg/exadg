/*
 * driver.cpp
 *
 *  Created on: 27.03.2020
 *      Author: fehn
 */

// likwid
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/incompressible_navier_stokes/driver.h>
#include <exadg/utilities/print_throughput.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm), pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
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
                           unsigned int const &                          refine_time,
                           bool const &                                  is_throughput_study)
{
  Timer timer;
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);
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

  application->create_grid(triangulation, refine_space, periodic_faces);
  print_grid_data(pcout, refine_space, *triangulation);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  application->set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);
  verify_boundary_conditions(*boundary_descriptor_velocity, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_pressure, *triangulation, periodic_faces);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);

  if(param.ale_formulation) // moving mesh
  {
    if(param.mesh_movement_type == MeshMovementType::Analytical)
    {
      std::shared_ptr<Function<dim>> mesh_motion = application->set_mesh_movement_function();

      moving_mesh.reset(new MovingMeshFunction<dim, Number>(
        *triangulation, mapping_degree, mapping_degree, mpi_comm, mesh_motion, param.start_time));
    }
    else if(param.mesh_movement_type == MeshMovementType::Poisson)
    {
      application->set_input_parameters_poisson(poisson_param);
      poisson_param.check_input_parameters();
      poisson_param.print(pcout, "List of input parameters for Poisson solver (moving mesh):");

      poisson_boundary_descriptor.reset(new Poisson::BoundaryDescriptor<1, dim>());
      application->set_boundary_conditions_poisson(poisson_boundary_descriptor);
      verify_boundary_conditions(*poisson_boundary_descriptor, *triangulation, periodic_faces);

      poisson_field_functions.reset(new Poisson::FieldFunctions<dim>());
      application->set_field_functions_poisson(poisson_field_functions);

      AssertThrow(poisson_param.right_hand_side == false,
                  ExcMessage("Poisson problem is used for mesh movement. Hence, "
                             "the right-hand side has to be zero for the Poisson problem."));

      // static mapping for Poisson solver is defined by poisson_param
      unsigned int const mapping_degree_poisson =
        get_mapping_degree(poisson_param.mapping, mapping_degree);
      poisson_mesh.reset(new Mesh<dim>(mapping_degree_poisson));

      // initialize Poisson operator
      poisson_operator.reset(new Poisson::Operator<dim, Number, dim>(*triangulation,
                                                                     poisson_mesh->get_mapping(),
                                                                     mapping_degree,
                                                                     periodic_faces,
                                                                     poisson_boundary_descriptor,
                                                                     poisson_field_functions,
                                                                     poisson_param,
                                                                     "Poisson",
                                                                     mpi_comm));

      // initialize matrix_free
      poisson_matrix_free_data.reset(new MatrixFreeData<dim, Number>());
      poisson_matrix_free_data->data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_partition;
      if(poisson_param.enable_cell_based_face_loops)
      {
        auto tria =
          std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation);
        Categorization::do_cell_based_loops(*tria, poisson_matrix_free_data->data);
      }
      poisson_operator->fill_matrix_free_data(*poisson_matrix_free_data);

      poisson_matrix_free.reset(new MatrixFree<dim, Number>());
      poisson_matrix_free->reinit(poisson_mesh->get_mapping(),
                                  poisson_matrix_free_data->get_dof_handler_vector(),
                                  poisson_matrix_free_data->get_constraint_vector(),
                                  poisson_matrix_free_data->get_quadrature_vector(),
                                  poisson_matrix_free_data->data);

      poisson_operator->setup(poisson_matrix_free, poisson_matrix_free_data);
      poisson_operator->setup_solver();

      moving_mesh.reset(
        new MovingMeshPoisson<dim, Number>(mapping_degree, mpi_comm, poisson_operator));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

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
  }
  else if(param.solver_type == SolverType::Steady)
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
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

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
  navier_stokes_operator->setup(matrix_free, matrix_free_data);

  if(!is_throughput_study)
  {
    // setup postprocessor
    postprocessor = application->construct_postprocessor(degree, mpi_comm);
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
                                                 refine_time,
                                                 mpi_comm,
                                                 postprocessor,
                                                 moving_mesh,
                                                 matrix_free));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting,
                                                       param,
                                                       refine_time,
                                                       mpi_comm,
                                                       postprocessor,
                                                       moving_mesh,
                                                       matrix_free));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
      {
        time_integrator.reset(
          new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction,
                                        param,
                                        refine_time,
                                        mpi_comm,
                                        postprocessor,
                                        moving_mesh,
                                        matrix_free));
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
        time_integrator->get_scaling_factor_time_derivative_term(),
        time_integrator->get_velocity());
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
  }

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  // move the mesh and update dependent data structures
  Timer timer;
  timer.restart();

  Timer sub_timer;

  sub_timer.restart();
  moving_mesh->move_mesh(time_integrator->get_next_time());
  timer_tree.insert({"Incompressible flow", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  matrix_free->update_mapping(moving_mesh->get_mapping());
  timer_tree.insert({"Incompressible flow", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  navier_stokes_operator->update_after_mesh_movement();
  timer_tree.insert({"Incompressible flow", "ALE", "Update operator"}, sub_timer.wall_time());

  sub_timer.restart();
  time_integrator->ale_update();
  timer_tree.insert({"Incompressible flow", "ALE", "Update time integrator"},
                    sub_timer.wall_time());

  timer_tree.insert({"Incompressible flow", "ALE"}, timer.wall_time());
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
      while(not time_integrator->finished())
      {
        time_integrator->advance_one_timestep_pre_solve(true);

        ale_update();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      }
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
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_statistics(double const total_time) const
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

    this->time_integrator->print_iterations();
  }

  // Wall times
  timer_tree.insert({"Incompressible flow"}, total_time);

  if(param.solver_type == SolverType::Unsteady)
  {
    timer_tree.insert({"Incompressible flow"}, time_integrator->get_timings());
  }
  else
  {
    timer_tree.insert({"Incompressible flow"}, driver_steady->get_timings());
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs            = navier_stokes_operator->get_number_of_dofs();
  unsigned int const            N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  if(param.solver_type == SolverType::Unsteady)
  {
    unsigned int N_time_steps = time_integrator->get_number_of_time_steps();
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
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(unsigned int const  degree,
                                    std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  (void)degree;

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

  Timer global_timer;
  global_timer.restart();
  Utilities::MPI::MinMaxAvg global_time;
  double                    wall_time = std::numeric_limits<double>::max();

  do
  {
    for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
    {
      Timer timer;
      timer.restart();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("degree_" + std::to_string(degree)).c_str());
#endif

      // apply matrix-vector product several times
      for(unsigned int i = 0; i < n_repetitions_inner; ++i)
      {
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
      }

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("degree_" + std::to_string(degree)).c_str());
#endif

      MPI_Barrier(mpi_comm);
      Utilities::MPI::MinMaxAvg wall_time_inner =
        Utilities::MPI::min_max_avg(timer.wall_time(), mpi_comm);

      wall_time = std::min(wall_time, wall_time_inner.avg / (double)n_repetitions_inner);
    }

    MPI_Barrier(mpi_comm);
    global_time = Utilities::MPI::min_max_avg(global_timer.wall_time(), mpi_comm);
  } while(global_time.avg < 1.0 /*wall time in seconds*/);

  types::global_dof_index dofs      = 0;
  unsigned int            fe_degree = 1;

  if(operator_type == Operator::CoupledNonlinearResidual ||
     operator_type == Operator::CoupledLinearized)
  {
    dofs = navier_stokes_operator->get_dof_handler_u().n_dofs() +
           navier_stokes_operator->get_dof_handler_p().n_dofs();

    fe_degree = navier_stokes_operator->get_polynomial_degree();
  }
  else if(operator_type == Operator::ConvectiveOperator ||
          operator_type == Operator::VelocityConvDiffOperator ||
          operator_type == Operator::HelmholtzOperator ||
          operator_type == Operator::ProjectionOperator ||
          operator_type == Operator::InverseMassMatrix)
  {
    dofs = navier_stokes_operator->get_dof_handler_u().n_dofs();

    fe_degree = navier_stokes_operator->get_polynomial_degree();
  }
  else if(operator_type == Operator::PressurePoissonOperator)
  {
    dofs = navier_stokes_operator->get_dof_handler_p().n_dofs();

    fe_degree = param.get_degree_p(navier_stokes_operator->get_polynomial_degree());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double throughput = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << throughput << std::endl
        << "DoFs/(sec*core): " << throughput/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(fe_degree, dofs, throughput);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace IncNS
} // namespace ExaDG
