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

#ifndef INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_

namespace ExaDG
{
namespace Poisson
{
/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Spatial discretization method
 */
enum class SpatialDiscretization
{
  Undefined,
  DG,
  CG
};

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
  Undefined,
  None,
  PointJacobi,
  BlockJacobi,
  AMG,
  Multigrid
};

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// currently no enum parameters here

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_ */
