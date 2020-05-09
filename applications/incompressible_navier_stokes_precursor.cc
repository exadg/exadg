/*
 * incompressible_navier_stokes_precursor.cc
 *
 *  Created on: 2017
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/parameter_handler.h>

// TODO: this function will be included in deal.II
#include "../include/utilities/parse_input.h"

// driver
#include "../include/incompressible_navier_stokes/driver_precursor.h"

// application
#include "incompressible_navier_stokes_test_cases/backward_facing_step/backward_facing_step.h"
#include "incompressible_navier_stokes_test_cases/fda/fda_nozzle_benchmark.h"
#include "incompressible_navier_stokes_test_cases/template_precursor/template_precursor.h"

class ApplicationSelector
{
public:
  template<int dim, typename Number>
  void
  add_parameters(dealii::ParameterHandler & prm, std::string name_of_application = "")
  {
    // application is unknown -> only add name of application to parameters
    if(name_of_application.length() == 0)
    {
      this->add_name_parameter(prm);
    }
    else // application is known -> add also application-specific parameters
    {
      name = name_of_application;
      this->add_name_parameter(prm);

      std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> app;
      if(name == "TemplatePrecursor")
        app.reset(new IncNS::TemplatePrecursor::Application<dim, Number>());
      else if(name == "BackwardFacingStep")
        app.reset(new IncNS::BackwardFacingStep::Application<dim, Number>());
      else if(name == "FDA")
        app.reset(new IncNS::FDA::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

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

private:
  void
  add_name_parameter(ParameterHandler & prm)
  {
    prm.enter_subsection("Application");
    prm.add_parameter("Name", name, "Name of application.");
    prm.leave_subsection();
  }

  std::string name = "MyApp";
};

struct Study
{
  Study()
  {
  }

  Study(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);

    parse_input(input_file, prm, true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",   precision,     "Floating point precision.",                    Patterns::Selection("float|double"));
      prm.add_parameter("Dim",         dim,           "Number of space dimension.",                   Patterns::Integer(2,3));
      prm.add_parameter("Degree",      degree,        "Polynomial degree of shape functions.",        Patterns::Integer(1,15));
      prm.add_parameter("RefineSpace", refine_space,  "Number of global, uniform mesh refinements.",  Patterns::Integer(0,20));
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree = 3;

  unsigned int refine_space = 0;
};

void
create_input_file(std::string const & name_of_application = "")
{
  dealii::ParameterHandler prm;

  Study study;
  study.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;

  ApplicationSelector selector;
  selector.add_parameters<Dim, Number>(prm, name_of_application);

  prm.print_parameters(std::cout,
                       dealii::ParameterHandler::JSON | dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    MPI_Comm const &    mpi_comm)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<IncNS::DriverPrecursor<dim, Number>> driver;
  driver.reset(new IncNS::DriverPrecursor<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  driver->setup(application, degree, refine_space);

  driver->solve();

  driver->print_statistics(timer.wall_time());
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  // check if parameter file is provided

  // ./incompressible_navier_stokes_precursor
  AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

  // ./incompressible_navier_stokes_precursor --help
  if(argc == 2 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      create_input_file();

    return 0;
  }
  // ./incompressible_navier_stokes_precursor --help NameOfApplication
  else if(argc == 3 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      create_input_file(argv[2]);

    return 0;
  }

  // the second argument is the input-file
  // ./incompressible_navier_stokes_precursor InputFile
  std::string input_file = std::string(argv[1]);
  Study       study(input_file);

  // run the simulation
  if(study.dim == 2 && study.precision == "float")
    run<2, float>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 2 && study.precision == "double")
    run<2, double>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 3 && study.precision == "float")
    run<3, float>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 3 && study.precision == "double")
    run<3, double>(input_file, study.degree, study.refine_space, mpi_comm);
  else
    AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));

  return 0;
}
