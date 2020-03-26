/*
 * compressible_navier_stokes.cc
 *
 *  Created on: 2018
 *      Author: fehn
 */

// driver
#include "../include/compressible_navier_stokes/driver.h"

// infrastructure for convergence studies
#include "../include/functionalities/convergence_study.h"

// applications

// template
#include "compressible_navier_stokes_test_cases/template/template.h"

// Euler equations
#include "compressible_navier_stokes_test_cases/euler_vortex/euler_vortex.h"

// Navier-Stokes equations
#include "compressible_navier_stokes_test_cases/couette/couette.h"
#include "compressible_navier_stokes_test_cases/flow_past_cylinder/flow_past_cylinder.h"
#include "compressible_navier_stokes_test_cases/manufactured_solution/manufactured_solution.h"
#include "compressible_navier_stokes_test_cases/poiseuille/poiseuille.h"
#include "compressible_navier_stokes_test_cases/shear_flow/shear_flow.h"
#include "compressible_navier_stokes_test_cases/taylor_green/taylor_green.h"
#include "compressible_navier_stokes_test_cases/turbulent_channel/turbulent_channel.h"

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

      std::shared_ptr<CompNS::ApplicationBase<dim, Number>> app;
      if(name == "Template")
        app.reset(new CompNS::Template::Application<dim, Number>());
      else if(name == "EulerVortex")
        app.reset(new CompNS::EulerVortex::Application<dim, Number>());
      else if(name == "Couette")
        app.reset(new CompNS::Couette::Application<dim, Number>());
      else if(name == "Poiseuille")
        app.reset(new CompNS::Poiseuille::Application<dim, Number>());
      else if(name == "ShearFlow")
        app.reset(new CompNS::ShearFlow::Application<dim, Number>());
      else if(name == "ManufacturedSolution")
        app.reset(new CompNS::ManufacturedSolution::Application<dim, Number>());
      else if(name == "FlowPastCylinder")
        app.reset(new CompNS::FlowPastCylinder::Application<dim, Number>());
      else if(name == "TaylorGreen")
        app.reset(new CompNS::TaylorGreen::Application<dim, Number>());
      else if(name == "TurbulentChannel")
        app.reset(new CompNS::TurbulentChannel::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

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
  std::shared_ptr<CompNS::Driver<dim, Number>> solver;
  solver.reset(new CompNS::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space, refine_time);

  solver->solve();

  solver->analyze_computing_times();
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // check if parameter file is provided

    // ./compressible_navier_stokes
    AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

    // ./compressible_navier_stokes --help
    if(argc == 2 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file();

      return 0;
    }
    // ./compressible_navier_stokes --help NameOfApplication
    else if(argc == 3 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file(argv[2]);

      return 0;
    }

    // the second argument is the input-file
    // ./compressible_navier_stokes InputFile
    std::string      input_file = std::string(argv[1]);
    ConvergenceStudy study(input_file);

    // k-refinement
    for(unsigned int degree = study.degree_min; degree <= study.degree_max; ++degree)
    {
      // h-refinement
      for(unsigned int refine_space = study.refine_space_min;
          refine_space <= study.refine_space_max;
          ++refine_space)
      {
        // dt-refinement
        for(unsigned int refine_time = study.refine_time_min; refine_time <= study.refine_time_max;
            ++refine_time)
        {
          // run the simulation
          if(study.dim == 2 && study.precision == "float")
            run<2, float>(input_file, degree, refine_space, refine_time, mpi_comm);
          else if(study.dim == 2 && study.precision == "double")
            run<2, double>(input_file, degree, refine_space, refine_time, mpi_comm);
          else if(study.dim == 3 && study.precision == "float")
            run<3, float>(input_file, degree, refine_space, refine_time, mpi_comm);
          else if(study.dim == 3 && study.precision == "double")
            run<3, double>(input_file, degree, refine_space, refine_time, mpi_comm);
          else
            AssertThrow(false,
                        ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
        }
      }
    }
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
