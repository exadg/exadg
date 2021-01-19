/*
 * solver.cpp
 *
 *  Created on: Nov 6, 2018
 *      Author: fehn
 */

// ExaDG
#include <exadg/incompressible_flow_with_transport/solver.h>

// applications
#include "applications/template/template.h"

// passive scalar
#include "applications/cavity/cavity.h"
#include "applications/lung/lung.h"

// natural convection (active scalar)
#include "applications/cavity_natural_convection/cavity_natural_convection.h"
#include "applications/mantle_convection/mantle_convection.h"
#include "applications/rayleigh_benard/rayleigh_benard.h"
#include "applications/rising_bubble/rising_bubble.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<FTI::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<FTI::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new FTI::Template::Application<dim, Number>(input_file));
    else if(name == "Cavity")
      app.reset(new FTI::Cavity::Application<dim, Number>(input_file));
    else if(name == "Lung")
      app.reset(new FTI::Lung::Application<dim, Number>(input_file));
    else if(name == "CavityNaturalConvection")
      app.reset(new FTI::CavityNaturalConvection::Application<dim, Number>(input_file));
    else if(name == "RayleighBenard")
      app.reset(new FTI::RayleighBenard::Application<dim, Number>(input_file));
    else if(name == "RisingBubble")
      app.reset(new FTI::RisingBubble::Application<dim, Number>(input_file));
    else if(name == "MantleConvection")
      app.reset(new FTI::MantleConvection::Application<dim, Number>(input_file));
    else
      AssertThrow(false, ExcMessage("This application does not exist!"));

    return app;
  }

  template<int dim, typename Number>
  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & input_file)
  {
    // if application is known, also add application-specific parameters
    try
    {
      std::shared_ptr<FTI::ApplicationBase<dim, Number>> app =
        get_application<dim, Number>(input_file);

      add_name_parameter(prm);
      app->add_parameters(prm);
    }
    catch(...) // if application is unknown, only add name of application to parameters
    {
      add_name_parameter(prm);
    }
  }

private:
  void
  add_name_parameter(ParameterHandler & prm)
  {
    prm.enter_subsection("Application");
    prm.add_parameter("Name", name, "Name of application.");
    prm.leave_subsection();
  }

  void
  parse_name_parameter(std::string input_file)
  {
    dealii::ParameterHandler prm;
    add_name_parameter(prm);
    prm.parse_input(input_file, "", true, true);
  }

  std::string name = "MyApp";
};

template<int dim, typename Number>
std::shared_ptr<FTI::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<FTI::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  return application;
}

template<int dim, typename Number>
void
add_parameters_application(dealii::ParameterHandler & prm, std::string const & input_file)
{
  ApplicationSelector selector;
  selector.add_parameters<dim, Number>(prm, input_file);
}

} // namespace ExaDG
