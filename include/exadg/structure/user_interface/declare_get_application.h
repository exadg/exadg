/*
 * declare_get_application.h
 *
 *  Created on: Dec 7, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_

#include <exadg/structure/user_interface/application_base.h>

namespace ExaDG
{
namespace Structure
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<ApplicationBase<dim, Number>>
get_application(std::string input_file, MPI_Comm const & comm);

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_DECLARE_GET_APPLICATION_H_ */
