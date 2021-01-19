/*
 * solver.cpp
 *
 *  Created on: 2018
 *      Author: fehn
 */

#include <exadg/compressible_navier_stokes/solver.h>

// applications
#include "applications/template/template.h"

// Euler equations
#include "applications/euler_vortex/euler_vortex.h"

// Navier-Stokes equations
#include "applications/couette/couette.h"
#include "applications/flow_past_cylinder/flow_past_cylinder.h"
#include "applications/manufactured_solution/manufactured_solution.h"
#include "applications/poiseuille/poiseuille.h"
#include "applications/shear_flow/shear_flow.h"
#include "applications/taylor_green/taylor_green.h"
#include "applications/turbulent_channel/turbulent_channel.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<CompNS::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new CompNS::Template::Application<dim, Number>(input_file));
    else if(name == "EulerVortex")
      app.reset(new CompNS::EulerVortex::Application<dim, Number>(input_file));
    else if(name == "Couette")
      app.reset(new CompNS::Couette::Application<dim, Number>(input_file));
    else if(name == "Poiseuille")
      app.reset(new CompNS::Poiseuille::Application<dim, Number>(input_file));
    else if(name == "ShearFlow")
      app.reset(new CompNS::ShearFlow::Application<dim, Number>(input_file));
    else if(name == "ManufacturedSolution")
      app.reset(new CompNS::ManufacturedSolution::Application<dim, Number>(input_file));
    else if(name == "FlowPastCylinder")
      app.reset(new CompNS::FlowPastCylinder::Application<dim, Number>(input_file));
    else if(name == "TaylorGreen")
      app.reset(new CompNS::TaylorGreen::Application<dim, Number>(input_file));
    else if(name == "TurbulentChannel")
      app.reset(new CompNS::TurbulentChannel::Application<dim, Number>(input_file));
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
      std::shared_ptr<CompNS::ApplicationBase<dim, Number>> app =
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
std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
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
