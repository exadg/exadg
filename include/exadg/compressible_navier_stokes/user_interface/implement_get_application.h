/*
 * implement_get_application.h
 *
 *  Created on: Dec 7, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_IMPLEMENT_GET_APPLICATION_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_IMPLEMENT_GET_APPLICATION_H_

#include <exadg/compressible_navier_stokes/user_interface/application_base.h>

namespace ExaDG
{
template<int dim, typename Number>
std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
get_application(std::string input_file, MPI_Comm const & comm)
{
  return std::make_shared<CompNS::Application<dim, Number>>(input_file, comm);
}

} // namespace ExaDG



#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_IMPLEMENT_GET_APPLICATION_H_ */
