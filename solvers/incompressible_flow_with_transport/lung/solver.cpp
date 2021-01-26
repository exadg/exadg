// ExaDG
#include <exadg/incompressible_flow_with_transport/solver.h>

// application
#include "lung.h"

namespace ExaDG
{
template<int dim, typename Number>
std::shared_ptr<FTI::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::shared_ptr<FTI::ApplicationBase<dim, Number>>(
    new FTI::Application<dim, Number>(input_file));
}
}