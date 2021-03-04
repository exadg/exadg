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

#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(std::string const & input_file,
                            MPI_Comm const &    comm,
                            bool const          is_test)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    partitioned_iterations({0, 0})
{
  dealii::ParameterHandler prm;

  add_parameters(prm, fsi_data);

  prm.parse_input(input_file, "", true, true);
}

template<int dim, typename Number>
void
Driver<dim, Number>::add_parameters(dealii::ParameterHandler & prm, PartitionedData & fsi_data)
{
  // clang-format off
  prm.enter_subsection("FSI");
    prm.add_parameter("Method",
                      fsi_data.method,
                      "Acceleration method.",
                      Patterns::Selection("Aitken|IQN-ILS|IQN-IMVLS"),
                      true);
    prm.add_parameter("AbsTol",
                      fsi_data.abs_tol,
                      "Absolute solver tolerance.",
                      Patterns::Double(0.0,1.0),
                      true);
    prm.add_parameter("RelTol",
                      fsi_data.rel_tol,
                      "Relative solver tolerance.",
                      Patterns::Double(0.0,1.0),
                      true);
    prm.add_parameter("OmegaInit",
                      fsi_data.omega_init,
                      "Initial relaxation parameter.",
                      Patterns::Double(0.0,1.0),
                      true);
    prm.add_parameter("ReusedTimeSteps",
                      fsi_data.reused_time_steps,
                      "Number of time steps reused for acceleration.",
                      Patterns::Integer(0, 100),
                      false);
    prm.add_parameter("PartitionedIterMax",
                      fsi_data.partitioned_iter_max,
                      "Maximum number of fixed-point iterations.",
                      Patterns::Integer(1,1000),
                      true);
    prm.add_parameter("GeometricTolerance",
                      fsi_data.geometric_tolerance,
                      "Tolerance used to locate points at FSI interface.",
                      Patterns::Double(0.0, 1.0),
                      false);
  prm.leave_subsection();
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const                            degree_fluid,
                           unsigned int const                            degree_structure,
                           unsigned int const                            refine_space_fluid,
                           unsigned int const                            refine_space_structure)

{
  Timer timer;
  timer.restart();

  print_exadg_header(pcout);
  pcout << "Setting up fluid-structure interaction solver:" << std::endl;

  if(not(is_test))
  {
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }
  print_MPI_info(pcout, mpi_comm);

  application = app;

  /**************************************** STRUCTURE *****************************************/
  {
    Timer timer_local;
    timer_local.restart();

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

    // triangulation and mapping
    unsigned int const mapping_degree_structure =
      get_mapping_degree(structure_param.mapping, degree_structure);
    application->create_grid_structure(structure_triangulation,
                                       structure_periodic_faces,
                                       refine_space_structure,
                                       structure_mapping,
                                       mapping_degree_structure);
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

    // field functions
    structure_field_functions.reset(new Structure::FieldFunctions<dim>());
    application->set_field_functions_structure(structure_field_functions);

    // setup spatial operator
    structure_operator.reset(new Structure::Operator<dim, Number>(*structure_triangulation,
                                                                  *structure_mapping,
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
    structure_matrix_free->reinit(*structure_mapping,
                                  structure_matrix_free_data->get_dof_handler_vector(),
                                  structure_matrix_free_data->get_constraint_vector(),
                                  structure_matrix_free_data->get_quadrature_vector(),
                                  structure_matrix_free_data->data);

    structure_operator->setup(structure_matrix_free, structure_matrix_free_data);

    // initialize postprocessor
    structure_postprocessor =
      application->construct_postprocessor_structure(degree_structure, mpi_comm);
    structure_postprocessor->setup(structure_operator->get_dof_handler(), *structure_mapping);

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

  /**************************************** STRUCTURE *****************************************/


  /****************************************** FLUID *******************************************/

  {
    Timer timer_local;
    timer_local.restart();

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

    // triangulation and mapping
    unsigned int const mapping_degree_fluid = get_mapping_degree(fluid_param.mapping, degree_fluid);
    application->create_grid_fluid(fluid_triangulation,
                                   fluid_periodic_faces,
                                   refine_space_fluid,
                                   fluid_static_mapping,
                                   mapping_degree_fluid);
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

    // ALE
    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      application->set_input_parameters_ale(ale_poisson_param);
      ale_poisson_param.check_input_parameters();
      AssertThrow(ale_poisson_param.right_hand_side == false,
                  ExcMessage("Parameter does not make sense in context of FSI."));
      AssertThrow(ale_poisson_param.mapping == fluid_param.mapping,
                  ExcMessage("Fluid and ALE must use the same mapping degree."));
      ale_poisson_param.print(pcout, "List of input parameters for ALE solver (Poisson):");

      ale_poisson_boundary_descriptor.reset(new Poisson::BoundaryDescriptor<1, dim>());
      application->set_boundary_conditions_ale(ale_poisson_boundary_descriptor);
      verify_boundary_conditions(*ale_poisson_boundary_descriptor,
                                 *fluid_triangulation,
                                 fluid_periodic_faces);

      ale_poisson_field_functions.reset(new Poisson::FieldFunctions<dim>());
      application->set_field_functions_ale(ale_poisson_field_functions);

      // initialize Poisson operator
      ale_poisson_operator.reset(
        new Poisson::Operator<dim, Number, dim>(*fluid_triangulation,
                                                *fluid_static_mapping,
                                                mapping_degree_fluid,
                                                fluid_periodic_faces,
                                                ale_poisson_boundary_descriptor,
                                                ale_poisson_field_functions,
                                                ale_poisson_param,
                                                "Poisson",
                                                mpi_comm));
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      application->set_input_parameters_ale(ale_elasticity_param);
      ale_elasticity_param.check_input_parameters();
      AssertThrow(ale_elasticity_param.body_force == false,
                  ExcMessage("Parameter does not make sense in context of FSI."));
      AssertThrow(ale_elasticity_param.mapping == fluid_param.mapping,
                  ExcMessage("Fluid and ALE must use the same mapping degree."));
      ale_elasticity_param.print(pcout, "List of input parameters for ALE solver (elasticity):");

      // boundary conditions
      ale_elasticity_boundary_descriptor.reset(new Structure::BoundaryDescriptor<dim>());
      application->set_boundary_conditions_ale(ale_elasticity_boundary_descriptor);
      verify_boundary_conditions(*ale_elasticity_boundary_descriptor,
                                 *fluid_triangulation,
                                 fluid_periodic_faces);

      // material_descriptor
      ale_elasticity_material_descriptor.reset(new Structure::MaterialDescriptor);
      application->set_material_ale(*ale_elasticity_material_descriptor);

      // field functions
      ale_elasticity_field_functions.reset(new Structure::FieldFunctions<dim>());
      application->set_field_functions_ale(ale_elasticity_field_functions);

      // setup spatial operator
      ale_elasticity_operator.reset(
        new Structure::Operator<dim, Number>(*fluid_triangulation,
                                             *fluid_static_mapping,
                                             mapping_degree_fluid,
                                             fluid_periodic_faces,
                                             ale_elasticity_boundary_descriptor,
                                             ale_elasticity_field_functions,
                                             ale_elasticity_material_descriptor,
                                             ale_elasticity_param,
                                             "ale_elasticity",
                                             mpi_comm));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    // initialize matrix_free
    ale_matrix_free_data.reset(new MatrixFreeData<dim, Number>());
    ale_matrix_free_data->data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::partition_partition;

    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      if(ale_poisson_param.enable_cell_based_face_loops)
      {
        auto tria = std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(
          fluid_triangulation);
        Categorization::do_cell_based_loops(*tria, ale_matrix_free_data->data);
      }
      ale_poisson_operator->fill_matrix_free_data(*ale_matrix_free_data);
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      ale_elasticity_operator->fill_matrix_free_data(*ale_matrix_free_data);
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    ale_matrix_free.reset(new MatrixFree<dim, Number>());
    ale_matrix_free->reinit(*fluid_static_mapping,
                            ale_matrix_free_data->get_dof_handler_vector(),
                            ale_matrix_free_data->get_constraint_vector(),
                            ale_matrix_free_data->get_quadrature_vector(),
                            ale_matrix_free_data->data);

    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      ale_poisson_operator->setup(ale_matrix_free, ale_matrix_free_data);
      ale_poisson_operator->setup_solver();
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      ale_elasticity_operator->setup(ale_matrix_free, ale_matrix_free_data);
      ale_elasticity_operator->setup_solver();
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    // mapping for fluid problem (moving mesh)
    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      fluid_moving_mapping.reset(
        new MovingMeshPoisson<dim, Number>(fluid_static_mapping, ale_poisson_operator));
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      fluid_moving_mapping.reset(new MovingMeshElasticity<dim, Number>(fluid_static_mapping,
                                                                       ale_elasticity_operator,
                                                                       ale_elasticity_param));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    fluid_mapping = fluid_moving_mapping;

    // initialize fluid_operator
    if(this->fluid_param.temporal_discretization ==
       IncNS::TemporalDiscretization::BDFCoupledSolution)
    {
      fluid_operator_coupled.reset(
        new IncNS::OperatorCoupled<dim, Number>(*fluid_triangulation,
                                                *fluid_mapping,
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
        new IncNS::OperatorDualSplitting<dim, Number>(*fluid_triangulation,
                                                      *fluid_mapping,
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
        new IncNS::OperatorPressureCorrection<dim, Number>(*fluid_triangulation,
                                                           *fluid_mapping,
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
    fluid_matrix_free->reinit(*fluid_mapping,
                              fluid_matrix_free_data->get_dof_handler_vector(),
                              fluid_matrix_free_data->get_constraint_vector(),
                              fluid_matrix_free_data->get_quadrature_vector(),
                              fluid_matrix_free_data->data);

    // setup Navier-Stokes operator
    fluid_operator->setup(fluid_matrix_free, fluid_matrix_free_data);

    // setup postprocessor
    fluid_postprocessor = application->construct_postprocessor_fluid(degree_fluid, mpi_comm);
    fluid_postprocessor->setup(*fluid_operator);

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  /****************************************** FLUID *******************************************/


  /*********************************** INTERFACE COUPLING *************************************/

  // structure to ALE
  {
    Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> ALE ..." << std::endl;

    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      std::vector<unsigned int> quad_indices;
      if(ale_poisson_param.spatial_discretization == Poisson::SpatialDiscretization::DG)
        quad_indices.emplace_back(ale_poisson_operator->get_quad_index());
      else if(ale_poisson_param.spatial_discretization == Poisson::SpatialDiscretization::CG)
        quad_indices.emplace_back(ale_poisson_operator->get_quad_index_gauss_lobatto());
      else
        AssertThrow(false, ExcMessage("not implemented."));

      VectorType displacement_structure;
      structure_operator->initialize_dof_vector(displacement_structure);
      structure_to_ale.reset(new InterfaceCoupling<dim, dim, Number>());
      structure_to_ale->setup(ale_matrix_free,
                              ale_poisson_operator->get_dof_index(),
                              quad_indices,
                              ale_poisson_boundary_descriptor->dirichlet_mortar_bc,
                              structure_triangulation,
                              structure_operator->get_dof_handler(),
                              *structure_mapping,
                              displacement_structure,
                              fsi_data.geometric_tolerance);
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      std::vector<unsigned int> quad_indices;
      quad_indices.emplace_back(ale_elasticity_operator->get_quad_index_gauss_lobatto());

      VectorType displacement_structure;
      structure_operator->initialize_dof_vector(displacement_structure);
      structure_to_ale.reset(new InterfaceCoupling<dim, dim, Number>());
      structure_to_ale->setup(ale_matrix_free,
                              ale_elasticity_operator->get_dof_index(),
                              quad_indices,
                              ale_elasticity_boundary_descriptor->dirichlet_mortar_bc,
                              structure_triangulation,
                              structure_operator->get_dof_handler(),
                              *structure_mapping,
                              displacement_structure,
                              fsi_data.geometric_tolerance);
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> ALE"}, timer_local.wall_time());
  }

  // structure to fluid
  {
    Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> fluid ..." << std::endl;

    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_linear());
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_nonlinear());
    quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_gauss_lobatto());

    VectorType velocity_structure;
    structure_operator->initialize_dof_vector(velocity_structure);
    structure_to_fluid.reset(new InterfaceCoupling<dim, dim, Number>());
    structure_to_fluid->setup(fluid_matrix_free,
                              fluid_operator->get_dof_index_velocity(),
                              quad_indices,
                              fluid_boundary_descriptor_velocity->dirichlet_mortar_bc,
                              structure_triangulation,
                              structure_operator->get_dof_handler(),
                              *structure_mapping,
                              velocity_structure,
                              fsi_data.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> fluid"}, timer_local.wall_time());
  }

  // fluid to structure
  {
    Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling fluid -> structure ..." << std::endl;

    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(structure_operator->get_quad_index());

    VectorType stress_fluid;
    fluid_operator->initialize_vector_velocity(stress_fluid);
    fluid_to_structure.reset(new InterfaceCoupling<dim, dim, Number>());
    fluid_to_structure->setup(structure_matrix_free,
                              structure_operator->get_dof_index(),
                              quad_indices,
                              structure_boundary_descriptor->neumann_mortar_bc,
                              fluid_triangulation,
                              fluid_operator->get_dof_handler_u(),
                              *fluid_mapping,
                              stress_fluid,
                              fsi_data.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling fluid -> structure"}, timer_local.wall_time());
  }

  /*********************************** INTERFACE COUPLING *************************************/


  /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

  // fluid
  {
    Timer timer_local;
    timer_local.restart();

    // setup time integrator before calling setup_solvers (this is necessary since the setup
    // of the solvers depends on quantities such as the time_step_size or gamma0!!!)
    AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
                ExcMessage("Invalid parameter in context of fluid-structure interaction."));

    // initialize fluid_operator
    if(this->fluid_param.temporal_discretization ==
       IncNS::TemporalDiscretization::BDFCoupledSolution)
    {
      fluid_time_integrator.reset(new IncNS::TimeIntBDFCoupled<dim, Number>(fluid_operator_coupled,
                                                                            fluid_param,
                                                                            0 /* refine_time */,
                                                                            mpi_comm,
                                                                            not(is_test),
                                                                            fluid_postprocessor,
                                                                            fluid_moving_mapping,
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
                                                        not(is_test),
                                                        fluid_postprocessor,
                                                        fluid_moving_mapping,
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
                                                             not(is_test),
                                                             fluid_postprocessor,
                                                             fluid_moving_mapping,
                                                             fluid_matrix_free));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    fluid_time_integrator->setup(fluid_param.restarted_simulation);

    fluid_operator->setup_solvers(fluid_time_integrator->get_scaling_factor_time_derivative_term(),
                                  fluid_time_integrator->get_velocity());

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  // Structure
  {
    Timer timer_local;
    timer_local.restart();

    // initialize time integrator
    structure_time_integrator.reset(
      new Structure::TimeIntGenAlpha<dim, Number>(structure_operator,
                                                  structure_postprocessor,
                                                  0 /* refine_time */,
                                                  structure_param,
                                                  mpi_comm,
                                                  not(is_test)));
    structure_time_integrator->setup(structure_param.restarted_simulation);

    structure_operator->setup_solver();

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

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
  bool const print_solver_info = fluid_time_integrator->print_solver_info();
  fluid_moving_mapping->update(fluid_time_integrator->get_next_time(),
                               print_solver_info,
                               not(is_test));
  timer_tree.insert({"FSI", "ALE", "Solve and reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_matrix_free->update_mapping(*fluid_mapping);
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
Driver<dim, Number>::coupling_structure_to_ale(VectorType const & displacement_structure) const
{
  Timer sub_timer;
  sub_timer.restart();

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
Driver<dim, Number>::apply_dirichlet_neumann_scheme(VectorType &       d_tilde,
                                                    VectorType const & d,
                                                    unsigned int       iteration) const
{
  coupling_structure_to_ale(d);

  // move the fluid mesh and update dependent data structures
  solve_ale();

  // update velocity boundary condition for fluid
  coupling_structure_to_fluid(iteration == 0);

  // solve fluid problem
  fluid_time_integrator->advance_one_timestep_partitioned_solve(iteration == 0, true);

  // update stress boundary condition for solid
  coupling_fluid_to_structure();

  // solve structural problem
  structure_time_integrator->advance_one_timestep_partitioned_solve(iteration == 0, true);

  d_tilde = structure_time_integrator->get_displacement_np();
}

template<int dim, typename Number>
bool
Driver<dim, Number>::check_convergence(VectorType const & residual) const
{
  double const residual_norm = residual.l2_norm();
  double const ref_norm_abs  = std::sqrt(structure_operator->get_number_of_dofs());
  double const ref_norm_rel  = structure_time_integrator->get_velocity_np().l2_norm() *
                              structure_time_integrator->get_time_step_size();

  bool const converged = (residual_norm < fsi_data.abs_tol * ref_norm_abs) ||
                         (residual_norm < fsi_data.rel_tol * ref_norm_rel);

  return converged;
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_solver_info_header(unsigned int const iteration) const
{
  if(fluid_time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "======================================================================" << std::endl
          << " Partitioned FSI: iteration counter = " << std::left << std::setw(8) << iteration
          << std::endl
          << "======================================================================" << std::endl;
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_solver_info_converged(unsigned int const iteration) const
{
  if(fluid_time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "Partitioned FSI iteration converged in " << iteration << " iterations." << std::endl;
  }
}

template<int dim, typename Number>
unsigned int
Driver<dim, Number>::solve_partitioned_problem() const
{
  // iteration counter
  unsigned int k = 0;

  // fixed-point iteration with dynamic relaxation (Aitken relaxation)
  if(fsi_data.method == "Aitken")
  {
    VectorType r_old, d;
    structure_operator->initialize_dof_vector(r_old);
    structure_operator->initialize_dof_vector(d);

    bool   converged = false;
    double omega     = 1.0;
    while(not converged and k < fsi_data.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure_time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure_time_integrator->get_displacement_np();

      VectorType d_tilde(d);
      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      VectorType r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        Timer timer;
        timer.restart();

        if(k == 0)
        {
          omega = fsi_data.omega_init;
        }
        else
        {
          VectorType delta_r = r;
          delta_r.add(-1.0, r_old);
          omega *= -(r_old * delta_r) / delta_r.norm_sqr();
        }

        r_old = r;

        d.add(omega, r);
        structure_time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "Aitken"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }
  }
  else if(fsi_data.method == "IQN-ILS")
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D.reset(new std::vector<VectorType>());
    R.reset(new std::vector<VectorType>());

    VectorType d, d_tilde, d_tilde_old, r, r_old;
    structure_operator->initialize_dof_vector(d);
    structure_operator->initialize_dof_vector(d_tilde);
    structure_operator->initialize_dof_vector(d_tilde_old);
    structure_operator->initialize_dof_vector(r);
    structure_operator->initialize_dof_vector(r_old);

    unsigned int const q = fsi_data.reused_time_steps;
    unsigned int const n = fluid_time_integrator->get_number_of_time_steps();

    bool converged = false;
    while(not(converged) and k < fsi_data.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure_time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure_time_integrator->get_displacement_np();

      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        Timer timer;
        timer.restart();

        if(k == 0 and (q == 0 or n == 0))
        {
          d.add(fsi_data.omega_init, r);
        }
        else
        {
          if(k >= 1)
          {
            // append D, R matrices
            VectorType delta_d_tilde = d_tilde;
            delta_d_tilde.add(-1.0, d_tilde_old);
            D->push_back(delta_d_tilde);

            VectorType delta_r = r;
            delta_r.add(-1.0, r_old);
            R->push_back(delta_r);
          }

          // fill vectors (including reuse)
          std::vector<VectorType> Q = *R;
          for(auto R_q : R_history)
            for(auto delta_r : *R_q)
              Q.push_back(delta_r);
          std::vector<VectorType> D_all = *D;
          for(auto D_q : D_history)
            for(auto delta_d : *D_q)
              D_all.push_back(delta_d);

          AssertThrow(D_all.size() == Q.size(), ExcMessage("D, Q vectors must have same size."));

          unsigned int const k_all = Q.size();
          if(k_all >= 1)
          {
            // compute QR-decomposition
            Matrix<Number> U(k_all);
            compute_QR_decomposition(Q, U);

            std::vector<Number> rhs(k_all, 0.0);
            for(unsigned int i = 0; i < k_all; ++i)
              rhs[i] = -Number(Q[i] * r);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(k_all, 0.0);
            backward_substitution(U, alpha, rhs);

            // d_{k+1} = d_tilde_{k} + delta d_tilde
            d = d_tilde;
            for(unsigned int i = 0; i < k_all; ++i)
              d.add(alpha[i], D_all[i]);
          }
          else // despite reuse, the vectors might be empty
          {
            d.add(fsi_data.omega_init, r);
          }
        }

        d_tilde_old = d_tilde;
        r_old       = r;

        structure_time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "IQN-ILS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }

    Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > q)
      D_history.erase(D_history.begin());
    if(R_history.size() > q)
      R_history.erase(R_history.begin());

    timer_tree.insert({"FSI", "IQN-ILS"}, timer.wall_time());
  }
  else if(fsi_data.method == "IQN-IMVLS")
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D.reset(new std::vector<VectorType>());
    R.reset(new std::vector<VectorType>());

    std::vector<VectorType> B;

    VectorType d, d_tilde, d_tilde_old, r, r_old, b, b_old;
    structure_operator->initialize_dof_vector(d);
    structure_operator->initialize_dof_vector(d_tilde);
    structure_operator->initialize_dof_vector(d_tilde_old);
    structure_operator->initialize_dof_vector(r);
    structure_operator->initialize_dof_vector(r_old);
    structure_operator->initialize_dof_vector(b);
    structure_operator->initialize_dof_vector(b_old);

    std::shared_ptr<Matrix<Number>> U;
    std::vector<VectorType>         Q;

    unsigned int const q = fsi_data.reused_time_steps;
    unsigned int const n = fluid_time_integrator->get_number_of_time_steps();

    bool converged = false;
    while(not converged and k < fsi_data.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure_time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure_time_integrator->get_displacement_np();

      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        Timer timer;
        timer.restart();

        // compute b vector
        inv_jacobian_times_residual(b, D_history, R_history, Z_history, r);

        if(k == 0 and (q == 0 or n == 0))
        {
          d.add(fsi_data.omega_init, r);
        }
        else
        {
          d = d_tilde;
          d.add(-1.0, b);

          if(k >= 1)
          {
            // append D, R, B matrices
            VectorType delta_d_tilde = d_tilde;
            delta_d_tilde.add(-1.0, d_tilde_old);
            D->push_back(delta_d_tilde);

            VectorType delta_r = r;
            delta_r.add(-1.0, r_old);
            R->push_back(delta_r);

            VectorType delta_b = delta_d_tilde;
            delta_b.add(1.0, b_old);
            delta_b.add(-1.0, b);
            B.push_back(delta_b);

            // compute QR-decomposition
            U.reset(new Matrix<Number>(k));
            Q = *R;
            compute_QR_decomposition(Q, *U);

            std::vector<Number> rhs(k, 0.0);
            for(unsigned int i = 0; i < k; ++i)
              rhs[i] = -Number(Q[i] * r);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(k, 0.0);
            backward_substitution(*U, alpha, rhs);

            for(unsigned int i = 0; i < k; ++i)
              d.add(alpha[i], B[i]);
          }
        }

        d_tilde_old = d_tilde;
        r_old       = r;
        b_old       = b;

        structure_time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "IQN-IMVLS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }

    Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > q)
      D_history.erase(D_history.begin());
    if(R_history.size() > q)
      R_history.erase(R_history.begin());

    // compute Z and add to Z_history
    std::shared_ptr<std::vector<VectorType>> Z;
    Z.reset(new std::vector<VectorType>());
    *Z = Q; // make sure that Z has correct size
    backward_substitution_multiple_rhs(*U, *Z, Q);
    Z_history.push_back(Z);
    if(Z_history.size() > q)
      Z_history.erase(Z_history.begin());

    timer_tree.insert({"FSI", "IQN-IMVLS"}, timer.wall_time());
  }
  else
  {
    AssertThrow(false, ExcMessage("This method is not implemented."));
  }

  return k;
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // The fluid domain is the master that dictates when the time loop is finished
  while(!fluid_time_integrator->finished())
  {
    // pre-solve
    fluid_time_integrator->advance_one_timestep_pre_solve(true);
    structure_time_integrator->advance_one_timestep_pre_solve(false);

    // solve (using strongly-coupled partitioned scheme)
    unsigned int iterations = solve_partitioned_problem();

    partitioned_iterations.first += 1;
    partitioned_iterations.second += iterations;

    print_solver_info_converged(iterations);

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
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  names = {"Partitioned iterations"};
  iterations_avg.resize(1);
  iterations_avg[0] =
    (double)partitioned_iterations.second / std::max(1.0, (double)partitioned_iterations.first);

  print_list_of_iterations(pcout, names, iterations_avg);
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

  // Average number of iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  print_partitioned_iterations();

  pcout << std::endl << "Fluid:" << std::endl;
  fluid_time_integrator->print_iterations();

  pcout << std::endl << "ALE:" << std::endl;
  fluid_moving_mapping->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure_time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid_time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, structure_time_integrator->get_timings(), "Structure");

  if(not(is_test))
  {
    pcout << std::endl << "Timings for level 1:" << std::endl;
    timer_tree.print_level(pcout, 1);

    pcout << std::endl << "Timings for level 2:" << std::endl;
    timer_tree.print_level(pcout, 2);

    // Throughput in DoFs/s per time step per core
    types::global_dof_index DoFs =
      fluid_operator->get_number_of_dofs() + structure_operator->get_number_of_dofs();

    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      DoFs += ale_poisson_operator->get_number_of_dofs();
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      DoFs += ale_elasticity_operator->get_number_of_dofs();
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

    Utilities::MPI::MinMaxAvg total_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
    double const              total_time_avg  = total_time_data.avg;

    unsigned int N_time_steps = fluid_time_integrator->get_number_of_time_steps();

    print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

    // computational costs in CPUh
    print_costs(pcout, total_time_avg, N_mpi_processes);
  }

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
