/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_UTILITIES_EXCEPTIONS_H_
#define INCLUDE_EXADG_UTILITIES_EXCEPTIONS_H_

#include <deal.II/base/exceptions.h>

namespace ExaDG
{
/**
 * The following exception is based on and motivated by dealii::ExcNotImplemented.
 */
DeclExceptionMsg(ExcNotImplemented,
                 "You are trying to use functionality in ExaDG that is "
                 "currently not implemented. In many cases, this indicates "
                 "that there simply didn't appear much of a need for it, or "
                 "that the author of the original code did not have the "
                 "time to implement a particular case. If you hit this "
                 "exception, it is therefore worth the time to look into "
                 "the code to find out whether you may be able to "
                 "implement the missing functionality. If you do, please "
                 "consider providing a patch to the ExaDG development "
                 "sources (see the ExaDG Wiki page on how to contribute).");

} // namespace ExaDG



#endif /* INCLUDE_EXADG_UTILITIES_EXCEPTIONS_H_ */
