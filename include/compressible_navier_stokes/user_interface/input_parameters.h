/*
 * input_parameters.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

#include "../../functionalities/enum_types.h"
#include "../../functionalities/print_functions.h"
#include "../../functionalities/restart_data.h"
#include "../../functionalities/solver_info_data.h"

#include "enum_types.h"

namespace CompNS
{
class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters();

  void
  check_input_parameters();

  void
  print(ConditionalOStream & pcout, std::string const & name);

private:
  void
  print_parameters_mathematical_model(ConditionalOStream & pcout);

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout);

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout);

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout);

  void
  print_parameters_solver(ConditionalOStream & pcout);

  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout);

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

  // triangulation type
  TriangulationType triangulation_type;

  // Type of mapping (polynomial degree) use for geometry approximation
  MappingType mapping;

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
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_*/
