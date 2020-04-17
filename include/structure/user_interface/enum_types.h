/*
 * enum_types.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_STRUCTURE_USER_INTERFACE_ENUM_TYPES_H_

// C/C++
#include <string>


#include "../../functionalities/enum_types.h"

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

std::string
enum_to_string(ProblemType const enum_type);

enum class Type2D
{
  Undefined,
  PlainStress,
  PlainStrain
};

std::string
enum_to_string(Type2D const enum_type);

enum class MaterialType
{
  Undefined,
  StVenantKirchhoff
};

std::string
enum_to_string(MaterialType const enum_type);

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
  CG
};

std::string
enum_to_string(Solver const enum_type);

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

std::string
enum_to_string(Preconditioner const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace Structure

#endif
