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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_FLUID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_FLUID_H_

// grid
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/grid/mapping_deformation_poisson.h>
#include <exadg/grid/mapping_deformation_structure.h>

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

// utilities
#include <exadg/utilities/timer_tree.h>

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class SolverFluid
{
public:
  SolverFluid()
  {
    timer_tree = std::make_shared<TimerTree>();
  }

  void
  setup(std::shared_ptr<FluidFSI::ApplicationBase<dim, Number>> application,
        MPI_Comm const                                          mpi_comm,
        bool const                                              is_test);

  void
  solve_ale() const;

  std::shared_ptr<TimerTree>
  get_timings_ale() const;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  // spatial discretization
  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> pde_operator;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> postprocessor;

  // ALE mapping
  std::shared_ptr<DeformedMappingBase<dim, Number>> ale_mapping;

  // ALE helper functions required by fluid time integrator
  std::shared_ptr<HelpersALE<Number>> helpers_ale;

  /*
   * Computation time (wall clock time).
   */
  std::shared_ptr<TimerTree> timer_tree;
};

template<int dim, typename Number>
void
SolverFluid<dim, Number>::setup(std::shared_ptr<FluidFSI::ApplicationBase<dim, Number>> application,
                                MPI_Comm const                                          mpi_comm,
                                bool const                                              is_test)
{
  // ALE: create grid motion object
  if(application->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
  {
    ale_mapping = std::make_shared<Poisson::DeformedMapping<dim, Number>>(
      application->get_grid(),
      application->get_mapping(),
      application->get_boundary_descriptor_ale_poisson(),
      application->get_field_functions_ale_poisson(),
      application->get_parameters_ale_poisson(),
      "Poisson",
      mpi_comm);
  }
  else if(application->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Elasticity)
  {
    ale_mapping = std::make_shared<Structure::DeformedMapping<dim, Number>>(
      application->get_grid(),
      application->get_mapping(),
      application->get_boundary_descriptor_ale_elasticity(),
      application->get_field_functions_ale_elasticity(),
      application->get_material_descriptor_ale_elasticity(),
      application->get_parameters_ale_elasticity(),
      "ale_elasticity",
      mpi_comm);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  // initialize pde_operator
  pde_operator = IncNS::create_operator<dim, Number>(application->get_grid(),
                                                     ale_mapping,
                                                     application->get_boundary_descriptor(),
                                                     application->get_field_functions(),
                                                     application->get_parameters(),
                                                     "fluid",
                                                     mpi_comm);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  if(application->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                        matrix_free_data->data);
  matrix_free->reinit(*ale_mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup Navier-Stokes operator
  pde_operator->setup(matrix_free, matrix_free_data);

  // setup postprocessor
  postprocessor = application->create_postprocessor();
  postprocessor->setup(*pde_operator);

  // setup time integrator before calling setup_solvers (this is necessary since the setup
  // of the solvers depends on quantities such as the time_step_size or gamma0!)
  AssertThrow(application->get_parameters().solver_type == IncNS::SolverType::Unsteady,
              dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // initialize time_integrator
  helpers_ale = std::make_shared<HelpersALE<Number>>();

  bool print_solver_info = is_test ? false : time_integrator->print_solver_info();
  helpers_ale->move_grid = [&](double const & time) {
    ale_mapping->update(time, print_solver_info, this->time_integrator->get_number_of_time_steps());
  };

  helpers_ale->update_pde_operator_after_grid_motion = [&]() {
    matrix_free->update_mapping(*ale_mapping);

    pde_operator->update_after_grid_motion();
  };

  time_integrator = IncNS::create_time_integrator<dim, Number>(
    pde_operator, helpers_ale, postprocessor, application->get_parameters(), mpi_comm, is_test);

  time_integrator->setup(application->get_parameters().restarted_simulation);

  pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                              time_integrator->get_velocity());
}

template<int dim, typename Number>
void
SolverFluid<dim, Number>::solve_ale() const
{
  dealii::Timer timer;
  timer.restart();

  dealii::Timer sub_timer;

  sub_timer.restart();
  helpers_ale->move_grid(time_integrator->get_next_time());
  timer_tree->insert({"ALE", "Solve and reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  helpers_ale->update_pde_operator_after_grid_motion();
  timer_tree->insert({"ALE", "Update matrix-free / PDE operator"}, sub_timer.wall_time());

  sub_timer.restart();
  time_integrator->ale_update();
  timer_tree->insert({"ALE", "Update time integrator"}, sub_timer.wall_time());

  timer_tree->insert({"ALE"}, timer.wall_time());
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
SolverFluid<dim, Number>::get_timings_ale() const
{
  return timer_tree;
}

} // namespace FSI
} // namespace ExaDG



#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SINGLE_FIELD_SOLVERS_FLUID_H_ */
