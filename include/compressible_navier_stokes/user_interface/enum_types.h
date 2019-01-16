/*
 * enum_types.h
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_

#include <string>

namespace CompNS
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., Euler euqations or Navier-Stokes equations
 */
enum class EquationType
{
  Undefined,
  Euler,
  NavierStokes
};

std::string
enum_to_string(EquationType const enum_type);

/*
 *  For energy boundary conditions, one can prescribe the temperature or the energy
 */
enum class EnergyBoundaryVariable
{
  Undefined,
  Energy,
  Temperature
};

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Temporal discretization method:
 *
 *    Explicit Runge-Kutta methods
 */
enum class TemporalDiscretization
{
  Undefined,
  ExplRK, // specify order of time integration scheme (order = stages)
  ExplRK3Stage4Reg2C,
  ExplRK3Stage7Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg2C,
  ExplRK4Stage8Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg3C,
  ExplRK5Stage9Reg2S,
  SSPRK // specify order and stages of time integration scheme
};

std::string
enum_to_string(TemporalDiscretization const enum_type);

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified,
  CFL,
  Diffusion,
  CFLAndDiffusion
};

std::string
enum_to_string(TimeStepCalculation const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/



/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/



/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/


} // namespace CompNS



#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_ */
