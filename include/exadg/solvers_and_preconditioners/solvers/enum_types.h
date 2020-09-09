/*
 * enum_types.h
 *
 *  Created on: Jun 13, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
namespace Elementwise
{
/*
 * Elementwise solver for block Jacobi preconditioner
 */
enum class Solver
{
  Undefined,
  CG,
  GMRES
};

std::string
enum_to_string(Solver const enum_type);

} // namespace Elementwise
} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_ENUM_TYPES_H_ */
