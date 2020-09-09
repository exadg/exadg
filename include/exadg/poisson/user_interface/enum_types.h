/*
 * enum_types.h
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

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

std::string
enum_to_string(SpatialDiscretization const enum_type);

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

std::string
enum_to_string(Solver const enum_type);

/*
 *  Preconditioner type for solution of linear system of equations
 */
enum class Preconditioner
{
  Undefined,
  None,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

std::string
enum_to_string(Preconditioner const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// currently no enum parameters here

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_ENUM_TYPES_H_ */
