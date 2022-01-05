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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class DriverFluid: public Driver <dim, Number>
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure) final;

  void
  solve() const final;

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
