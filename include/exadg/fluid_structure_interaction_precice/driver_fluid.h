/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

// grid
#include <exadg/grid/grid_motion_elasticity.h>
#include <exadg/grid/grid_motion_poisson.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/poisson/spatial_discretization/operator.h>

// IncNS
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class DriverFluid : public Driver<dim, Number>
{
private:
  using VectorType = typename LinearAlgebra::distributed::Vector<Number>;

public:
  DriverFluid(std::string const & input_file, MPI_Comm const & comm, bool const is_test)
    : Driver<dim, Number>(input_file, comm, is_test)
  {
  }

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure)
  {
    (void) degree_structure;
    (void) refine_space_structure;

    print_exadg_header(this->pcout);
    this->pcout << "Setting up coupled fluid solver:" << std::endl;

    if(!(this->is_test))
    {
      print_dealii_info(this->pcout);
      print_matrixfree_info<Number>(this->pcout);
    }
    print_MPI_info(this->pcout, this->mpi_comm);

    this->application = application;

    /****************************************** FLUID *******************************************/

    {
      // parameters fluid
      this->application->set_parameters_fluid(degree_fluid);
      this->application->get_parameters_fluid().check(this->pcout);
      this->application->get_parameters_fluid().print(
        this->pcout, "List of parameters for incompressible flow solver:");

      // Some FSI specific Asserts
      AssertThrow(this->application->get_parameters_fluid().problem_type ==
                    IncNS::ProblemType::Unsteady,
                  ExcMessage("Invalid parameter in context of fluid-structure interaction."));
      AssertThrow(this->application->get_parameters_fluid().ale_formulation == true,
                  ExcMessage("Invalid parameter in context of fluid-structure interaction."));

      // grid
      GridData fluid_grid_data;
      fluid_grid_data.triangulation_type =
        this->application->get_parameters_fluid().triangulation_type;
      fluid_grid_data.n_refine_global = refine_space_fluid;
      fluid_grid_data.mapping_degree =
        get_mapping_degree(this->application->get_parameters_fluid().mapping,
                           this->application->get_parameters_fluid().degree_u);

      fluid_grid = this->application->create_grid_fluid(fluid_grid_data);
      print_grid_info(this->pcout, *fluid_grid);

      // field functions and boundary conditions

      // fluid
      this->application->set_boundary_descriptor_fluid();
      IncNS::verify_boundary_conditions<dim>(this->application->get_boundary_descriptor_fluid(),
                                             *fluid_grid);

      this->application->set_field_functions_fluid();

      // ALE
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        this->application->set_parameters_ale_poisson(fluid_grid_data.mapping_degree);
        this->application->get_parameters_ale_poisson().check();
        AssertThrow(this->application->get_parameters_ale_poisson().right_hand_side == false,
                    ExcMessage("Parameter does not make sense in context of FSI."));
        AssertThrow(this->application->get_parameters_ale_poisson().mapping ==
                      this->application->get_parameters_fluid().mapping,
                    ExcMessage("Fluid and ALE must use the same mapping degree."));
        this->application->get_parameters_ale_poisson().print(
          this->pcout, "List of parameters for ALE solver (Poisson):");

        this->application->set_boundary_descriptor_ale_poisson();
        verify_boundary_conditions(*this->application->get_boundary_descriptor_ale_poisson(),
                                   *fluid_grid);

        this->application->set_field_functions_ale_poisson();

        // initialize Poisson operator
        ale_poisson_operator = std::make_shared<Poisson::Operator<dim, Number, dim>>(
          fluid_grid,
          this->application->get_boundary_descriptor_ale_poisson(),
          this->application->get_field_functions_ale_poisson(),
          this->application->get_parameters_ale_poisson(),
          "Poisson",
          this->mpi_comm);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        this->application->set_parameters_ale_elasticity(fluid_grid_data.mapping_degree);
        this->application->get_parameters_ale_elasticity().check();
        AssertThrow(this->application->get_parameters_ale_elasticity().body_force == false,
                    ExcMessage("Parameter does not make sense in context of FSI."));
        AssertThrow(this->application->get_parameters_ale_elasticity().mapping ==
                      this->application->get_parameters_fluid().mapping,
                    ExcMessage("Fluid and ALE must use the same mapping degree."));
        this->application->get_parameters_ale_elasticity().print(
          this->pcout, "List of parameters for ALE solver (elasticity):");

        // boundary conditions
        this->application->set_boundary_descriptor_ale_elasticity();
        verify_boundary_conditions(*this->application->get_boundary_descriptor_ale_elasticity(),
                                   *fluid_grid);

        // material_descriptor
        this->application->set_material_descriptor_ale_elasticity();

        // field functions
        this->application->set_field_functions_ale_elasticity();

        // setup spatial operator
        ale_elasticity_operator = std::make_shared<Structure::Operator<dim, Number>>(
          fluid_grid,
          this->application->get_boundary_descriptor_ale_elasticity(),
          this->application->get_field_functions_ale_elasticity(),
          this->application->get_material_descriptor_ale_elasticity(),
          this->application->get_parameters_ale_elasticity(),
          "ale_elasticity",
          this->mpi_comm);
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      // initialize matrix_free_data
      ale_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();

      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        if(this->application->get_parameters_ale_poisson().enable_cell_based_face_loops)
          Categorization::do_cell_based_loops(*fluid_grid->triangulation,
                                              ale_matrix_free_data->data);

        ale_matrix_free_data->append(ale_poisson_operator);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        ale_matrix_free_data->append(ale_elasticity_operator);
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      // initialize matrix_free
      ale_matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      ale_matrix_free->reinit(*fluid_grid->mapping,
                              ale_matrix_free_data->get_dof_handler_vector(),
                              ale_matrix_free_data->get_constraint_vector(),
                              ale_matrix_free_data->get_quadrature_vector(),
                              ale_matrix_free_data->data);

      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        ale_poisson_operator->setup(ale_matrix_free, ale_matrix_free_data);
        ale_poisson_operator->setup_solver();
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        ale_elasticity_operator->setup(ale_matrix_free, ale_matrix_free_data);
        ale_elasticity_operator->setup_solver();
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      // mapping for fluid problem (moving mesh)
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        fluid_grid_motion =
          std::make_shared<GridMotionPoisson<dim, Number>>(fluid_grid->get_static_mapping(),
                                                           ale_poisson_operator);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        fluid_grid_motion = std::make_shared<GridMotionElasticity<dim, Number>>(
          fluid_grid->get_static_mapping(),
          ale_elasticity_operator,
          this->application->get_parameters_ale_elasticity());
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      fluid_grid->attach_grid_motion(fluid_grid_motion);

      // initialize fluid_operator
      fluid_operator =
        IncNS::create_operator<dim, Number>(fluid_grid,
                                            this->application->get_boundary_descriptor_fluid(),
                                            this->application->get_field_functions_fluid(),
                                            this->application->get_parameters_fluid(),
                                            "fluid",
                                            this->mpi_comm);

      // initialize matrix_free
      fluid_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
      fluid_matrix_free_data->append(fluid_operator);

      fluid_matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      if(this->application->get_parameters_fluid().use_cell_based_face_loops)
        Categorization::do_cell_based_loops(*fluid_grid->triangulation,
                                            fluid_matrix_free_data->data);
      fluid_matrix_free->reinit(*fluid_grid->get_dynamic_mapping(),
                                fluid_matrix_free_data->get_dof_handler_vector(),
                                fluid_matrix_free_data->get_constraint_vector(),
                                fluid_matrix_free_data->get_quadrature_vector(),
                                fluid_matrix_free_data->data);

      // setup Navier-Stokes operator
      fluid_operator->setup(fluid_matrix_free, fluid_matrix_free_data);

      // setup postprocessor
      fluid_postprocessor = this->application->create_postprocessor_fluid();
      fluid_postprocessor->setup(*fluid_operator);
    }

    /****************************************** FLUID *******************************************/


    /*********************************** INTERFACE COUPLING *************************************/

    // structure to ALE

    // structure to fluid

    // fluid to structure

    /*********************************** INTERFACE COUPLING *************************************/


    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

    // fluid
    {
      // setup time integrator before calling setup_solvers (this is necessary since the setup
      // of the solvers depends on quantities such as the time_step_size or gamma0!!!)
      AssertThrow(this->application->get_parameters_fluid().solver_type ==
                    IncNS::SolverType::Unsteady,
                  ExcMessage("Invalid parameter in context of fluid-structure interaction."));

      // initialize fluid_time_integrator
      fluid_time_integrator =
        IncNS::create_time_integrator<dim, Number>(fluid_operator,
                                                   this->application->get_parameters_fluid(),
                                                   0 /* refine_time */,
                                                   this->mpi_comm,
                                                   this->is_test,
                                                   fluid_postprocessor);

      fluid_time_integrator->setup(this->application->get_parameters_fluid().restarted_simulation);

      fluid_operator->setup_solvers(
        fluid_time_integrator->get_scaling_factor_time_derivative_term(),
        fluid_time_integrator->get_velocity());
    }

    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/
  }

  void
  solve() const final
  {
    Assert(false, ExcNotImplemented());

    Assert(this->application->get_parameters_fluid().adaptive_time_stepping == false,
           ExcNotImplemented())
  }

private:
  /****************************************** FLUID *******************************************/

  // grid
  std::shared_ptr<Grid<dim, Number>> fluid_grid;

  // moving mapping (ALE)
  std::shared_ptr<GridMotionBase<dim, Number>> fluid_grid_motion;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> fluid_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     fluid_matrix_free;

  // spatial discretization
  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> fluid_operator;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> fluid_time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /************************************ ALE - MOVING MESH *************************************/

  // use a PDE solver for moving mesh problem
  std::shared_ptr<MatrixFreeData<dim, Number>> ale_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     ale_matrix_free;

  // Poisson-type mesh motion
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> ale_poisson_operator;

  // elasticity-type mesh motion
  std::shared_ptr<Structure::Operator<dim, Number>> ale_elasticity_operator;

  /************************************ ALE - MOVING MESH *************************************/
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_ */
