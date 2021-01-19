/*
 * solver.cpp
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

// ExaDG
#include <exadg/fluid_structure_interaction/solver.h>

// applications
#include "applications/bending_wall/bending_wall.h"
#include "applications/cylinder_with_flag/cylinder_with_flag.h"
#include "applications/pressure_wave/pressure_wave.h"
#include "applications/template/template.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<FSI::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<FSI::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new FSI::Template::Application<dim, Number>(input_file));
    else if(name == "CylinderWithFlag")
      app.reset(new FSI::CylinderWithFlag::Application<dim, Number>(input_file));
    else if(name == "BendingWall")
      app.reset(new FSI::BendingWall::Application<dim, Number>(input_file));
    else if(name == "PressureWave")
      app.reset(new FSI::PressureWave::Application<dim, Number>(input_file));
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
      std::shared_ptr<FSI::ApplicationBase<dim, Number>> app =
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
std::shared_ptr<FSI::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  ApplicationSelector selector;

  std::shared_ptr<FSI::ApplicationBase<dim, Number>> application =
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
