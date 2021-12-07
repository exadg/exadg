/*
 * declare_get_application.h
 *
 *  Created on: Dec 7, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POISSON_USER_INTERFACE_DECLARE_GET_APPLICATION_H_
#define INCLUDE_EXADG_POISSON_USER_INTERFACE_DECLARE_GET_APPLICATION_H_

#include <exadg/poisson/user_interface/application_base.h>

namespace ExaDG
{
namespace Poisson
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<ApplicationBase<dim, Number>>
get_application(std::string input_file, MPI_Comm const & comm);

} // namespace Poisson
} // namespace ExaDG



#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_DECLARE_GET_APPLICATION_H_ */
