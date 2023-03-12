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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
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

} // namespace Elementwise
} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ENUM_TYPES_H_ */
