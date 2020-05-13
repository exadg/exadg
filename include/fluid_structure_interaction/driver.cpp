/*
 * driver.cpp
 *
 *  Created on: 01.04.2020
 *      Author: fehn
 */

#include "driver.h"
#include "../utilities/print_throughput.h"

namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
    partitioned_iterations(0)
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
                           unsigned int const &                          degree_structure,
                           unsigned int const &                          refine_space_fluid,
                           unsigned int const &                          refine_space_structure)

{
  Timer timer;
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  /**************************************** STRUCTURE *****************************************/

  application->set_input_parameters_structure(structure_param);
  structure_param.check_input_parameters();
  // Some FSI specific Asserts
  AssertThrow(structure_param.pull_back_traction == true,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));
  structure_param.print(pcout, "List of input parameters for structure:");

  // triangulation
  if(structure_param.triangulation_type == TriangulationType::Distributed)
  {
    structure_triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(structure_param.triangulation_type == TriangulationType::FullyDistributed)
  {
    structure_triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  application->create_grid_structure(structure_triangulation,
                                     refine_space_structure,
                                     structure_periodic_faces);
  print_grid_data(pcout, refine_space_structure, *structure_triangulation);

  // boundary conditions
  structure_boundary_descriptor.reset(new Structure::BoundaryDescriptor<dim>());
  application->set_boundary_conditions_structure(structure_boundary_descriptor);
  verify_boundary_conditions(*structure_boundary_descriptor,
                             *structure_triangulation,
                             structure_periodic_faces);

  // material_descriptor
  structure_material_descriptor.reset(new Structure::MaterialDescriptor);
  application->set_material_structure(*structure_material_descriptor);

  // field functions and boundary conditions
  structure_field_functions.reset(new Structure::FieldFunctions<dim>());
  application->set_field_functions_structure(structure_field_functions);

  // mapping
  unsigned int const mapping_degree_structure =
    get_mapping_degree(structure_param.mapping, degree_structure);
  structure_mesh.reset(new Mesh<dim>(mapping_degree_structure));

  // setup spatial operator
  structure_operator.reset(new Structure::Operator<dim, Number>(*structure_triangulation,
                                                                structure_mesh->get_mapping(),
                                                                degree_structure,
                                                                structure_periodic_faces,
                                                                structure_boundary_descriptor,
                                                                structure_field_functions,
                                                                structure_material_descriptor,
                                                                structure_param,
                                                                "elasticity",
                                                                mpi_comm));

  // initialize matrix_free
  structure_matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  structure_matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  structure_operator->fill_matrix_free_data(*structure_matrix_free_data);

  structure_matrix_free.reset(new MatrixFree<dim, Number>());
  structure_matrix_free->reinit(structure_mesh->get_mapping(),
                                structure_matrix_free_data->get_dof_handler_vector(),
                                structure_matrix_free_data->get_constraint_vector(),
                                structure_matrix_free_data->get_quadrature_vector(),
                                structure_matrix_free_data->data);

  structure_operator->setup(structure_matrix_free, structure_matrix_free_data);

  // initialize postprocessor
  structure_postprocessor =
    application->construct_postprocessor_structure(degree_structure, mpi_comm);
  structure_postprocessor->setup(structure_operator->get_dof_handler(),
                                 structure_mesh->get_mapping());

  /**************************************** STRUCTURE *****************************************/


  /****************************************** FLUID *******************************************/

  // parameters fluid
  application->set_input_parameters_fluid(fluid_param);
  fluid_param.check_input_parameters(pcout);
  fluid_param.print(pcout, "List of input parameters for incompressible flow solver:");

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

  application->create_grid_fluid(fluid_triangulation, refine_space_fluid, fluid_periodic_faces);
  print_grid_data(pcout, refine_space_fluid, *fluid_triangulation);

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

  // Poisson
  AssertThrow(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson,
              ExcMessage("not implemented."));

  application->set_input_parameters_poisson(poisson_param);
  poisson_param.check_input_parameters();
  poisson_param.print(pcout, "List of input parameters for ALE solver (Poisson):");

  poisson_boundary_descriptor.reset(new Poisson::BoundaryDescriptor<1, dim>());
  application->set_boundary_conditions_poisson(poisson_boundary_descriptor);
  verify_boundary_conditions(*poisson_boundary_descriptor,
                             *fluid_triangulation,
                             fluid_periodic_faces);

  poisson_field_functions.reset(new Poisson::FieldFunctions<dim>());
  application->set_field_functions_poisson(poisson_field_functions);

  AssertThrow(poisson_param.right_hand_side == false,
              ExcMessage("Parameter does not make sense in context of FSI."));

  // mapping for Poisson solver (static mesh)
  unsigned int const mapping_degree_poisson =
    get_mapping_degree(poisson_param.mapping, degree_poisson);
  poisson_mesh.reset(new Mesh<dim>(mapping_degree_poisson));

  // initialize Poisson operator
  poisson_operator.reset(new Poisson::Operator<dim, Number, dim>(*fluid_triangulation,
                                                                 poisson_mesh->get_mapping(),
                                                                 degree_poisson,
                                                                 fluid_periodic_faces,
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
    auto tria = std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(
      fluid_triangulation);
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

  // structure to ALE
  {
    std::vector<unsigned int> quad_indices;
    if(poisson_param.spatial_discretization == Poisson::SpatialDiscretization::DG)
      quad_indices.emplace_back(poisson_operator->get_quad_index());
    else if(poisson_param.spatial_discretization == Poisson::SpatialDiscretization::CG)
      quad_indices.emplace_back(poisson_operator->get_quad_index_gauss_lobatto());
    else
      AssertThrow(false, ExcMessage("not implemented."));

    VectorType displacement_structure;
    structure_operator->initialize_dof_vector(displacement_structure);
    structure_to_ale.reset(new InterfaceCoupling<dim, dim, Number>(mpi_comm));
    structure_to_ale->setup(poisson_matrix_free,
                            poisson_operator->get_dof_index(),
                            quad_indices,
                            poisson_boundary_descriptor->dirichlet_mortar_bc,
                            structure_operator->get_dof_handler(),
                            structure_mesh->get_mapping(),
                            displacement_structure);
  }

  // mapping for fluid problem (moving mesh)
  unsigned int const mapping_degree_fluid = get_mapping_degree(fluid_param.mapping, degree_fluid);
  fluid_moving_mesh.reset(new MovingMeshPoisson<dim, Number>(
    mapping_degree_fluid, mpi_comm, poisson_operator, fluid_param.start_time));

  fluid_mesh = fluid_moving_mesh;

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_operator_coupled.reset(
      new IncNS::DGNavierStokesCoupled<dim, Number>(*fluid_triangulation,
                                                    fluid_mesh->get_mapping(),
                                                    degree_fluid,
                                                    fluid_periodic_faces,
                                                    fluid_boundary_descriptor_velocity,
                                                    fluid_boundary_descriptor_pressure,
                                                    fluid_field_functions,
                                                    fluid_param,
                                                    "fluid",
                                                    mpi_comm));

    fluid_operator = fluid_operator_coupled;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_operator_dual_splitting.reset(
      new IncNS::DGNavierStokesDualSplitting<dim, Number>(*fluid_triangulation,
                                                          fluid_mesh->get_mapping(),
                                                          degree_fluid,
                                                          fluid_periodic_faces,
                                                          fluid_boundary_descriptor_velocity,
                                                          fluid_boundary_descriptor_pressure,
                                                          fluid_field_functions,
                                                          fluid_param,
                                                          "fluid",
                                                          mpi_comm));

    fluid_operator = fluid_operator_dual_splitting;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_operator_pressure_correction.reset(
      new IncNS::DGNavierStokesPressureCorrection<dim, Number>(*fluid_triangulation,
                                                               fluid_mesh->get_mapping(),
                                                               degree_fluid,
                                                               fluid_periodic_faces,
                                                               fluid_boundary_descriptor_velocity,
                                                               fluid_boundary_descriptor_pressure,
                                                               fluid_field_functions,
                                                               fluid_param,
                                                               "fluid",
                                                               mpi_comm));

    fluid_operator = fluid_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  fluid_matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  fluid_matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  if(fluid_param.use_cell_based_face_loops)
  {
    auto tria = std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(
      fluid_triangulation);
    Categorization::do_cell_based_loops(*tria, fluid_matrix_free_data->data);
  }
  fluid_operator->fill_matrix_free_data(*fluid_matrix_free_data);

  fluid_matrix_free.reset(new MatrixFree<dim, Number>());
  fluid_matrix_free->reinit(fluid_mesh->get_mapping(),
                            fluid_matrix_free_data->get_dof_handler_vector(),
                            fluid_matrix_free_data->get_constraint_vector(),
                            fluid_matrix_free_data->get_quadrature_vector(),
                            fluid_matrix_free_data->data);

  // setup Navier-Stokes operator
  fluid_operator->setup(fluid_matrix_free, fluid_matrix_free_data);

  // setup postprocessor
  fluid_postprocessor = application->construct_postprocessor_fluid(degree_fluid, mpi_comm);
  fluid_postprocessor->setup(*fluid_operator);

  // structure to fluid
  {
    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_linear());
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_nonlinear());
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_gauss_lobatto());

    VectorType velocity_structure;
    structure_operator->initialize_dof_vector(velocity_structure);
    structure_to_fluid.reset(new InterfaceCoupling<dim, dim, Number>(mpi_comm));
    structure_to_fluid->setup(fluid_matrix_free,
                              fluid_operator->get_dof_index_velocity(),
                              quad_indices,
                              fluid_boundary_descriptor_velocity->dirichlet_mortar_bc,
                              structure_operator->get_dof_handler(),
                              structure_mesh->get_mapping(),
                              velocity_structure);
  }

  // fluid to structure
  {
    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(structure_operator->get_quad_index());

    VectorType stress_fluid;
    fluid_operator->initialize_vector_velocity(stress_fluid);
    fluid_to_structure.reset(new InterfaceCoupling<dim, dim, Number>(mpi_comm));
    fluid_to_structure->setup(structure_matrix_free,
                              structure_operator->get_dof_index(),
                              quad_indices,
                              structure_boundary_descriptor->neumann_mortar_bc,
                              fluid_operator->get_dof_handler_u(),
                              fluid_mesh->get_mapping(),
                              stress_fluid);
  }

  /****************************************** FLUID *******************************************/


  /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

  // fluid

  // setup time integrator before calling setup_solvers (this is necessary since the setup
  // of the solvers depends on quantities such as the time_step_size or gamma0!!!)
  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_time_integrator.reset(new IncNS::TimeIntBDFCoupled<dim, Number>(fluid_operator_coupled,
                                                                          fluid_param,
                                                                          0 /* refine_time */,
                                                                          mpi_comm,
                                                                          fluid_postprocessor,
                                                                          fluid_moving_mesh,
                                                                          fluid_matrix_free));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_time_integrator.reset(
      new IncNS::TimeIntBDFDualSplitting<dim, Number>(fluid_operator_dual_splitting,
                                                      fluid_param,
                                                      0 /* refine_time */,
                                                      mpi_comm,
                                                      fluid_postprocessor,
                                                      fluid_moving_mesh,
                                                      fluid_matrix_free));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_time_integrator.reset(
      new IncNS::TimeIntBDFPressureCorrection<dim, Number>(fluid_operator_pressure_correction,
                                                           fluid_param,
                                                           0 /* refine_time */,
                                                           mpi_comm,
                                                           fluid_postprocessor,
                                                           fluid_moving_mesh,
                                                           fluid_matrix_free));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  fluid_time_integrator->setup(fluid_param.restarted_simulation);

  fluid_operator->setup_solvers(fluid_time_integrator->get_scaling_factor_time_derivative_term(),
                                fluid_time_integrator->get_velocity());

  // Structure

  // initialize time integrator
  structure_time_integrator.reset(new Structure::TimeIntGenAlpha<dim, Number>(
    structure_operator, structure_postprocessor, 0 /* refine_time */, structure_param, mpi_comm));
  structure_time_integrator->setup(structure_param.restarted_simulation);

  structure_operator->setup_solver();

  /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

  timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // The fluid domain is the master that dictates the start time
  structure_time_integrator->reset_time(fluid_time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  // The fluid domain is the master that dictates the time step size
  structure_time_integrator->set_current_time_step_size(
    fluid_time_integrator->get_time_step_size());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_ale() const
{
  Timer timer;
  timer.restart();

  Timer sub_timer;

  sub_timer.restart();
  fluid_moving_mesh->move_mesh(fluid_time_integrator->get_next_time());
  timer_tree.insert({"FSI", "ALE", "Solve and reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_matrix_free->update_mapping(fluid_moving_mesh->get_mapping());
  timer_tree.insert({"FSI", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_operator->update_after_mesh_movement();
  timer_tree.insert({"FSI", "ALE", "Update operator"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_time_integrator->ale_update();
  timer_tree.insert({"FSI", "ALE", "Update time integrator"}, sub_timer.wall_time());

  timer_tree.insert({"FSI", "ALE"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_ale(VectorType & displacement_structure,
                                               bool const   extrapolate) const
{
  Timer sub_timer;

  sub_timer.restart();
  structure_operator->initialize_dof_vector(displacement_structure);
  if(extrapolate)
    structure_time_integrator->extrapolate_displacement_to_np(displacement_structure);
  else
    displacement_structure = structure_time_integrator->get_displacement_np();

  structure_to_ale->update_data(displacement_structure);
  timer_tree.insert({"FSI", "Coupling structure -> ALE"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_fluid(bool const extrapolate) const
{
  Timer sub_timer;
  sub_timer.restart();

  VectorType velocity_structure;
  structure_operator->initialize_dof_vector(velocity_structure);
  if(extrapolate)
    structure_time_integrator->extrapolate_velocity_to_np(velocity_structure);
  else
    velocity_structure = structure_time_integrator->get_velocity_np();

  structure_to_fluid->update_data(velocity_structure);

  timer_tree.insert({"FSI", "Coupling structure -> fluid"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_fluid_to_structure() const
{
  Timer sub_timer;
  sub_timer.restart();

  VectorType stress_fluid;
  fluid_operator->initialize_vector_velocity(stress_fluid);
  // calculate fluid stress at fluid-structure interface
  fluid_operator->interpolate_stress_bc(stress_fluid,
                                        fluid_time_integrator->get_velocity_np(),
                                        fluid_time_integrator->get_pressure_np());
  stress_fluid *= -1.0;
  fluid_to_structure->update_data(stress_fluid);

  timer_tree.insert({"FSI", "Coupling fluid -> structure"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // relaxation factor
  double const OMEGA_MAX        = 0.1;
  double       omega            = OMEGA_MAX;
  double const residual_scaling = std::sqrt(structure_operator->get_number_of_dofs());

  // The fluid domain is the master that dictates when the time loop is finished
  while(!fluid_time_integrator->finished())
  {
    // pre-solve
    fluid_time_integrator->advance_one_timestep_pre_solve();
    structure_time_integrator->advance_one_timestep_pre_solve();

    // TODO
    // strongly-coupled partitioned iteration:
    // fixed-point iteration with dynamic relaxation (Aitken relaxation)
    double const       TOL        = 1.e-6;
    unsigned int const N_ITER_MAX = 100;
    unsigned int       iter       = 0;
    bool               converged  = false;
    VectorType         residual_last;
    structure_operator->initialize_dof_vector(residual_last);
    while(not converged and iter < N_ITER_MAX)
    {
      VectorType displacement_last;
      coupling_structure_to_ale(displacement_last, iter == 0);

      // move the fluid mesh and update dependent data structures
      solve_ale();

      // update velocity boundary condition for fluid
      coupling_structure_to_fluid(iter == 0);

      // solve fluid problem
      fluid_time_integrator->advance_one_timestep_solve();

      // update stress boundary condition for solid
      coupling_fluid_to_structure();

      // solve structural problem
      structure_time_integrator->advance_one_timestep_solve();

      // compute residual and check convergence
      VectorType residual;
      structure_operator->initialize_dof_vector(residual);
      residual = structure_time_integrator->get_displacement_np();
      residual.add(-1.0, displacement_last);
      converged = (residual.l2_norm() < TOL * residual_scaling);

      // relaxation step
      if(not(converged))
      {
        if(iter == 0)
        {
          omega = OMEGA_MAX; // std::max(omega, OMEGA_MAX);
        }
        else
        {
          VectorType residual_increment;
          structure_operator->initialize_dof_vector(residual_increment);
          residual_increment = residual;
          residual_increment.add(-1.0, residual_last);
          double const norm_residual_increment_sqr = residual_increment.norm_sqr();
          double const inner_product               = residual_last * residual_increment;
          omega *= -inner_product / norm_residual_increment_sqr;
        }
        residual_last = residual;

        // TODO
        pcout << "omega = " << omega << std::endl;
        structure_time_integrator->relax_displacement(omega, displacement_last);
      }

      // increment counter of partitioned iteration
      ++iter;
    }

    partitioned_iterations += iter;

    // TODO
    pcout << "Partitioned iteration converged in " << iter << " iterations." << std::endl
          << std::endl;

    // post-solve
    fluid_time_integrator->advance_one_timestep_post_solve();
    structure_time_integrator->advance_one_timestep_post_solve();

    if(fluid_param.adaptive_time_stepping)
      synchronize_time_step_size();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_partitioned_iterations() const
{
  unsigned int const N_time_steps =
    std::max(1, (int)fluid_time_integrator->get_number_of_time_steps());

  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  names = {"Partitioned iterations"};
  iterations_avg.resize(1);
  iterations_avg[0] = (double)partitioned_iterations / (double)N_time_steps;

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  // print
  for(unsigned int i = 0; i < iterations_avg.size(); ++i)
  {
    pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
          << std::setprecision(2) << std::right << std::setw(6) << iterations_avg[i] << std::endl;
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_statistics(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // Average number of iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  print_partitioned_iterations();

  pcout << std::endl << "Fluid:" << std::endl;
  fluid_time_integrator->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure_time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid_time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, structure_time_integrator->get_timings(), "Structure");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // computational costs in CPUh
  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg total_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              total_time_avg  = total_time_data.avg;

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
