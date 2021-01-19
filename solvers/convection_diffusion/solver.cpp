/*
 * solver.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

// ExaDG
#include <exadg/convection_diffusion/solver.h>

// application
#include "applications/template/application.h"

// applications - convection
#include "applications/deforming_hill/deforming_hill.h"
#include "applications/rotating_hill/rotating_hill.h"
#include "applications/sine_wave/sine_wave.h"

// applications - diffusion
#include "applications/decaying_hill/decaying_hill.h"

// applications - convection-diffusion
#include "applications/boundary_layer/boundary_layer.h"
#include "applications/const_rhs_const_or_circular_wind/const_rhs.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new ConvDiff::Template::Application<dim, Number>(input_file));
    else if(name == "SineWave")
      app.reset(new ConvDiff::SineWave::Application<dim, Number>(input_file));
    else if(name == "DeformingHill")
      app.reset(new ConvDiff::DeformingHill::Application<dim, Number>(input_file));
    else if(name == "RotatingHill")
      app.reset(new ConvDiff::RotatingHill::Application<dim, Number>(input_file));
    else if(name == "DecayingHill")
      app.reset(new ConvDiff::DecayingHill::Application<dim, Number>(input_file));
    else if(name == "BoundaryLayer")
      app.reset(new ConvDiff::BoundaryLayer::Application<dim, Number>(input_file));
    else if(name == "ConstRHS")
      app.reset(new ConvDiff::ConstRHS::Application<dim, Number>(input_file));
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
      std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> app =
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
std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> application =
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
