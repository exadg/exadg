/*
 * enum_types.h
 *
 *  Created on: Jun 13, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_

#include <string>

namespace Elementwise
{
/*
 * Elementwise preconditioner for block Jacobi preconditioner (only relevant for
 * elementwise iterative solution procedure)
 */
enum class Preconditioner
{
  Undefined,
  None,
  InverseMassMatrix
};

std::string
enum_to_string(Preconditioner const enum_type);

} // namespace Elementwise


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_ */
