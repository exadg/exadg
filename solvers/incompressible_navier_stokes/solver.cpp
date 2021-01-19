/*
 * solver.cpp
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

// ExaDG
#include <exadg/incompressible_navier_stokes/solver.h>

// applications
#include "applications/template/template.h"

// Stokes flow - analytical solutions
#include "applications/stokes_curl_flow/stokes_curl_flow.h"
#include "applications/stokes_guermond/stokes_guermond.h"
#include "applications/stokes_shahbazi/stokes_shahbazi.h"

// Navier-Stokes flow - analytical solutions
#include "applications/beltrami/beltrami.h"
#include "applications/cavity/cavity.h"
#include "applications/couette/couette.h"
#include "applications/free_stream/free_stream.h"
#include "applications/kovasznay/kovasznay.h"
#include "applications/orr_sommerfeld/orr_sommerfeld.h"
#include "applications/poiseuille/poiseuille.h"
#include "applications/unstable_beltrami/unstable_beltrami.h"
#include "applications/vortex/vortex.h"
#include "applications/vortex_periodic/vortex_periodic.h"

// more complex applications and turbulence
#include "applications/fda/fda_nozzle_benchmark.h"
#include "applications/flow_past_cylinder/flow_past_cylinder.h"
#include "applications/kelvin_helmholtz/kelvin_helmholtz.h"
#include "applications/periodic_hill/periodic_hill.h"
#include "applications/shear_layer/shear_layer.h"
#include "applications/taylor_green_vortex/taylor_green_vortex.h"
#include "applications/tum/tum.h"
#include "applications/turbulent_channel/turbulent_channel.h"

// incompressible flow with scalar transport (but can also be used for pure fluid simulations)
#include <exadg/incompressible_flow_with_transport/user_interface/application_base.h>
#include "../incompressible_flow_with_transport/applications/lung/lung.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new IncNS::Template::Application<dim, Number>(input_file));
    else if(name == "StokesGuermond")
      app.reset(new IncNS::StokesGuermond::Application<dim, Number>(input_file));
    else if(name == "StokesShahbazi")
      app.reset(new IncNS::StokesShahbazi::Application<dim, Number>(input_file));
    else if(name == "StokesCurlFlow")
      app.reset(new IncNS::StokesCurlFlow::Application<dim, Number>(input_file));
    else if(name == "FreeStream")
      app.reset(new IncNS::FreeStream::Application<dim, Number>(input_file));
    else if(name == "Couette")
      app.reset(new IncNS::Couette::Application<dim, Number>(input_file));
    else if(name == "Poiseuille")
      app.reset(new IncNS::Poiseuille::Application<dim, Number>(input_file));
    else if(name == "Cavity")
      app.reset(new IncNS::Cavity::Application<dim, Number>(input_file));
    else if(name == "Kovasznay")
      app.reset(new IncNS::Kovasznay::Application<dim, Number>(input_file));
    else if(name == "VortexPeriodic")
      app.reset(new IncNS::VortexPeriodic::Application<dim, Number>(input_file));
    else if(name == "Vortex")
      app.reset(new IncNS::Vortex::Application<dim, Number>(input_file));
    else if(name == "OrrSommerfeld")
      app.reset(new IncNS::OrrSommerfeld::Application<dim, Number>(input_file));
    else if(name == "Beltrami")
      app.reset(new IncNS::Beltrami::Application<dim, Number>(input_file));
    else if(name == "UnstableBeltrami")
      app.reset(new IncNS::UnstableBeltrami::Application<dim, Number>(input_file));
    else if(name == "KelvinHelmholtz")
      app.reset(new IncNS::KelvinHelmholtz::Application<dim, Number>(input_file));
    else if(name == "ShearLayer")
      app.reset(new IncNS::ShearLayer::Application<dim, Number>(input_file));
    else if(name == "TUM")
      app.reset(new IncNS::TUM::Application<dim, Number>(input_file));
    else if(name == "FlowPastCylinder")
      app.reset(new IncNS::FlowPastCylinder::Application<dim, Number>(input_file));
    else if(name == "TaylorGreenVortex")
      app.reset(new IncNS::TaylorGreenVortex::Application<dim, Number>(input_file));
    else if(name == "TurbulentChannel")
      app.reset(new IncNS::TurbulentChannel::Application<dim, Number>(input_file));
    else if(name == "PeriodicHill")
      app.reset(new IncNS::PeriodicHill::Application<dim, Number>(input_file));
    else if(name == "FDA")
      app.reset(new IncNS::FDA::Application<dim, Number>(input_file));
    else if(name == "Lung")
      app.reset(new FTI::Lung::Application<dim, Number>(input_file));
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
      std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app =
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
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBase<dim, Number>> application =
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
