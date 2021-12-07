/*
 * declare_get_application.h
 *
 *  Created on: Dec 7, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_DECLARE_GET_APPLICATION_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_DECLARE_GET_APPLICATION_H_


#include <exadg/convection_diffusion/user_interface/application_base.h>

namespace ExaDG
{
namespace ConvDiff
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<ApplicationBase<dim, Number>>
get_application(std::string input_file, MPI_Comm const & comm);

} // namespace ConvDiff
} // namespace ExaDG


#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_DECLARE_GET_APPLICATION_H_ */
