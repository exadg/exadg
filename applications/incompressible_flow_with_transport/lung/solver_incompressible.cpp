// solver
#include <exadg/incompressible_navier_stokes/solver.h>

// application
#include "application.h"

namespace ExaDG
{
template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<FTI::Application<dim, Number>>(input_file);
}

} // namespace ExaDG
