/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_INPUT_PARAMETERS_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/acoustic_conservation_equations/user_interface/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/operators/inverse_mass_parameters.h>
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>

namespace ExaDG
{
namespace Acoustics
{
class Parameters
{
public:
  // standard constructor that initializes parameters
  Parameters();

  void
  check() const;

  void
  print(dealii::ConditionalOStream const & pcout, std::string const & name) const;

private:
  void
  print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const;

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  Formulation formulation;

  // if there are acoustic source terms, set right_hand_side = true
  bool right_hand_side;

  // Use the aero-acoustic source term that is internally computed from the fluid solution
  bool aero_acoustic_source_term;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // start time of simulation
  double start_time;

  // end time of simulation
  double end_time;

  // speed_of_sound of underlying fluid
  double speed_of_sound;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  TimeStepCalculation calculation_of_time_step_size;

  // cfl number: note that this cfl number is the first in a series of cfl numbers
  // when performing temporal convergence tests, i.e., cfl_real = cfl, cfl/2, cfl/4, ...
  double cfl;

  // dt = CFL/max(k_p,k_u)^{exp} * h / c
  double cfl_exponent_fe_degree;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // maximum number of time steps
  unsigned int max_number_of_time_steps;

  // number of refinements for temporal discretization
  unsigned int n_refine_time;

  // order of time integration scheme
  unsigned int order_time_integrator;

  // start time integrator with low order time integrator, i.e., first order Euler method
  bool start_with_low_order;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // use adaptive timestepping
  bool adaptive_time_stepping;

  // restart
  RestartData restart_data;

  // show solver performance (wall time, number of iterations)
  SolverInfoData solver_info_data;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Grid data
  GridData grid;

  // Mapping
  unsigned int mapping_degree;

  // Polynomial degree of velocity shape functions
  unsigned int degree_u;

  // Polynomial degree of pressure shape functions
  unsigned int degree_p;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_INPUT_PARAMETERS_H_ */
