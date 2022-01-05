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
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
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
    Assert(false, ExcNotImplemented());
    (void)application;
    (void)degree_fluid;
    (void)degree_structure;
    (void)refine_space_fluid;
    (void)refine_space_structure;
  }

  void
  solve() const final
  {
    Assert(false, ExcNotImplemented());

    Assert(this->application->get_parameters_fluid().adaptive_time_stepping == false, ExcNotImplemented())
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
