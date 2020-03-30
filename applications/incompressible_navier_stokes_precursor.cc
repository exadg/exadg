/*
 *unsteady_navier_stokes_two_domains.cc
 *
 *  Created on: 2017
 *      Author: fehn
 */

// specify the flow problem that has to be solved

// TODO
//#include "incompressible_navier_stokes_test_cases/template_two_domains.h"
//#include "incompressible_navier_stokes_test_cases/turbulent_channel_two_domains.h"

// driver
#include "../include/incompressible_navier_stokes/driver_precursor.h"

// infrastructure for convergence studies
#include "../include/functionalities/convergence_study.h"

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

void
create_input_file(std::string const & name_of_application = "")
{
  dealii::ParameterHandler prm;

  ConvergenceStudy study;
  study.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;

  ApplicationSelector selector;
  selector.add_parameters<Dim, Number>(prm, name_of_application);

  prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::JSON, false);
}

template<int dim, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm)
{
  std::shared_ptr<IncNS::DriverPrecursor<dim, Number>> driver;
  driver.reset(new IncNS::DriverPrecursor<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  driver->setup(application, degree, refine_space, refine_time);

  driver->solve();

  driver->analyze_computing_times();
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // check if parameter file is provided

    // ./incompressible_navier_stokes
    AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

    // ./incompressible_navier_stokes --help
    if(argc == 2 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file();

      return 0;
    }
    // ./incompressible_navier_stokes --help NameOfApplication
    else if(argc == 3 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file(argv[2]);

      return 0;
    }

    // the second argument is the input-file
    // ./incompressible_navier_stokes InputFile
    std::string      input_file = std::string(argv[1]);
    ConvergenceStudy study(input_file);

    AssertThrow(study.degree_min == study.degree_max &&
                  study.refine_space_min == study.refine_space_max &&
                  study.refine_time_min == study.refine_time_max,
                ExcMessage("Automatic convergence studies are currently not "
                           "implemented for precursor-type simulations."));

    // run the simulation
    unsigned int const degree = study.degree_min, refine_space = study.refine_space_min,
                       refine_time = study.refine_time_min;
    if(study.dim == 2 && study.precision == "float")
      run<2, float>(input_file, degree, refine_space, refine_time, mpi_comm);
    else if(study.dim == 2 && study.precision == "double")
      run<2, double>(input_file, degree, refine_space, refine_time, mpi_comm);
    else if(study.dim == 3 && study.precision == "float")
      run<3, float>(input_file, degree, refine_space, refine_time, mpi_comm);
    else if(study.dim == 3 && study.precision == "double")
      run<3, double>(input_file, degree, refine_space, refine_time, mpi_comm);
    else
      AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  return 0;
}
