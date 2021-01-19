/*
 * solver_precursor.cpp
 *
 *  Created on: 2017
 *      Author: fehn
 */

// ExaDG
#include <exadg/incompressible_navier_stokes/solver_precursor.h>

// applications
#include "applications/template_precursor/template_precursor.h"

#include "applications/backward_facing_step/backward_facing_step.h"
#include "applications/fda/fda_nozzle_benchmark.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> app;
    if(name == "TemplatePrecursor")
      app.reset(new IncNS::TemplatePrecursor::Application<dim, Number>(input_file));
    else if(name == "BackwardFacingStep")
      app.reset(new IncNS::BackwardFacingStep::Application<dim, Number>(input_file));
    else if(name == "FDA")
      app.reset(new IncNS::FDA::Application<dim, Number>(input_file));
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
      std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> app =
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
std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> application =
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
