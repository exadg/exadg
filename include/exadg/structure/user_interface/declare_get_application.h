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

#ifndef EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_
#define EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_

// ExaDG
#include <exadg/structure/user_interface/application_base.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
std::shared_ptr<ApplicationBase<dim, Number>>
get_application(std::string input_file, MPI_Comm const & comm);

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_ */
