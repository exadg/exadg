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

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_

// C/C++
#include <string>

namespace ExaDG
{
namespace Structure
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType describes whether a steady or an unsteady problem has to be solved
 */
enum class ProblemType
{
  Undefined,
  Steady,
  QuasiStatic,
  Unsteady
};

enum class Type2D
{
  Undefined,
  PlaneStress,
  PlaneStrain
};

enum class MaterialType
{
  Undefined,
  StVenantKirchhoff,
  IncompressibleNeoHookean
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

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *   Solver for linear system of equations
 */
enum class Solver
{
  Undefined,
  CG,
  FGMRES
};

/*
 *  Preconditioner type for solution of linear system of equations
 */
enum class Preconditioner
{
  None,
  PointJacobi,
  Multigrid,
  AMG
};

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace Structure
} // namespace ExaDG

#endif
