/*
 * enum_types.h
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_POISSON_USER_INTERFACE_ENUM_TYPES_H_

namespace Poisson
{
/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 * Triangulation type
 */
enum class TriangulationType
{
  Undefined,
  Distributed,
  FullyDistributed
};

std::string
enum_to_string(TriangulationType const enum_type);

/*
 *  Mapping type (polynomial degree)
 */
enum class MappingType
{
  Affine,
  Quadratic,
  Cubic,
  Isoparametric
};

std::string
enum_to_string(MappingType const enum_type);

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



#endif /* INCLUDE_POISSON_USER_INTERFACE_ENUM_TYPES_H_ */
