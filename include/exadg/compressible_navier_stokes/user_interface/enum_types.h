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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
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

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

/*
 *  QuadratureRule
 */
enum class QuadratureRule
{
  Standard,
  Overintegration32k,
  Overintegration2k
};


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
} // namespace ExaDG


#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_ */
