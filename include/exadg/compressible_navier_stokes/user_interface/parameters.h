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
#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

#include <exadg/compressible_navier_stokes/user_interface/enum_types.h>
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>
#include <exadg/utilities/print_functions.h>
#include "exadg/solvers_and_preconditioners/solvers/solver_data.h"

namespace ExaDG
{
namespace CompNS
{
class Parameters
{
public:
  // standard constructor that initializes parameters with default values
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

  void
  print_parameters_solver(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const;

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  EquationType equation_type;

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // start time of simulation
  double start_time;

  // end time of simulation
  double end_time;

  // dynamic viscosity
  double dynamic_viscosity;

  // reference density needed to calculate the kinematic viscosity from the specified
  // dynamic viscosity
  double reference_density;

  // heat_capacity_ratio
  double heat_capacity_ratio;

  // thermal conductivity
  double thermal_conductivity;

  // specific gas constant
  double specific_gas_constant;

  // maximum temperature (needed to calculate time step size according to CFL condition)
  double max_temperature;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // temporal discretization method
  TemporalDiscretization temporal_discretization;

  // order of time integration scheme
  unsigned int order_time_integrator;

  // number of Runge-Kutta stages
  unsigned int stages;

  // calculation of time step size
  TimeStepCalculation calculation_of_time_step_size;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // maximum number of time steps
  unsigned int max_number_of_time_steps;

  // number of refinements for temporal discretization
  unsigned int n_refine_time;

  // maximum velocity needed when calculating the time step according to cfl-condition
  double max_velocity;

  // cfl number
  double cfl_number;

  // diffusion number (relevant number for limitation of time step size
  // when treating the diffusive term explicitly)
  double diffusion_number;

  // exponent of fe_degree used in the calculation of the CFL time step size
  double exponent_fe_degree_cfl;

  // exponent of fe_degree used in the calculation of the diffusion time step size
  double exponent_fe_degree_viscous;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // Restart
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

  // polynomial degree of shape functions
  unsigned int degree;

  QuadratureRule n_q_points_convective, n_q_points_viscous;

  // diffusive term: Symmetric interior penalty Galerkin (SIPG) discretization
  // interior penalty parameter scaling factor: default value is 1.0
  double IP_factor;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                NUMERICAL PARAMETERS                                */
  /*                                                                                    */
  /**************************************************************************************/

  // detect instabilities (norm of solution vector grows by a large factor from one time
  // step to the next
  bool detect_instabilities;

  // use combined operator for viscous term and convective term in order to improve run
  // time
  bool use_combined_operator;

  /**************************************************************************************/
  /*                                                                                    */
  /*                            ELEMENTWISE INVERSE MASS                                */
  /*                                                                                    */
  /**************************************************************************************/
  // Used when matrix-free inverse mass operator is not available and when the spatial
  // discretization is DG, e.g. simplex.

  bool solve_elementwise_mass_system_matrix_free;

  SolverData solver_data_elementwise_inverse_mass;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_*/
