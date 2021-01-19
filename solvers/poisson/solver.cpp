/*
 * solver.cpp
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

// ExaDG
#include <exadg/poisson/solver.h>

// applications
#include "applications/template/template.h"

#include "applications/gaussian/gaussian.h"
#include "applications/lung/lung.h"
#include "applications/lung_tubus/lung_tubus.h"
#include "applications/nozzle/nozzle.h"
#include "applications/sine/sine.h"
#include "applications/slit/slit.h"
#include "applications/torus/torus.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<Poisson::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<Poisson::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new Poisson::Template::Application<dim, Number>(input_file));
    else if(name == "Gaussian")
      app.reset(new Poisson::Gaussian::Application<dim, Number>(input_file));
    else if(name == "Sine")
      app.reset(new Poisson::Sine::Application<dim, Number>(input_file));
    else if(name == "Slit")
      app.reset(new Poisson::Slit::Application<dim, Number>(input_file));
    else if(name == "Torus")
      app.reset(new Poisson::Torus::Application<dim, Number>(input_file));
    else if(name == "Nozzle")
      app.reset(new Poisson::Nozzle::Application<dim, Number>(input_file));
    else if(name == "LungTubus")
      app.reset(new Poisson::LungTubus::Application<dim, Number>(input_file));
    else if(name == "Lung")
      app.reset(new Poisson::Lung::Application<dim, Number>(input_file));
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
      std::shared_ptr<Poisson::ApplicationBase<dim, Number>> app =
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
std::shared_ptr<Poisson::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<Poisson::ApplicationBase<dim, Number>> application =
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
