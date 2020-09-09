/*
 * solver.cpp
 *
 *  Created on: 2018
 *      Author: fehn
 */

// driver
#include <exadg/compressible_navier_stokes/driver.h>

// utilities
#include <exadg/utilities/convergence_study.h>

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

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  ConvergenceStudy study;
  study.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;

  ApplicationSelector selector;
  selector.add_parameters<Dim, Number>(prm, input_file);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<CompNS::Driver<dim, Number>> solver;
  solver.reset(new CompNS::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space, refine_time);

  solver->solve();

  solver->print_statistics(timer.wall_time());
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1 or (argc == 2 and std::string(argv[1]) == "--help"))
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To create an input file, use: ./solver --create_input_file input_file"
                << std::endl;
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[argc - 1]);
  }

  if(argc == 3 and std::string(argv[1]) == "--create_input_file")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      ExaDG::create_input_file(input_file);

    return 0;
  }

  ExaDG::ConvergenceStudy study(input_file);

  // k-refinement
  for(unsigned int degree = study.degree_min; degree <= study.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = study.refine_space_min; refine_space <= study.refine_space_max;
        ++refine_space)
    {
      // dt-refinement
      for(unsigned int refine_time = study.refine_time_min; refine_time <= study.refine_time_max;
          ++refine_time)
      {
        // run the simulation
        if(study.dim == 2 && study.precision == "float")
          ExaDG::run<2, float>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 2 && study.precision == "double")
          ExaDG::run<2, double>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 3 && study.precision == "float")
          ExaDG::run<3, float>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 3 && study.precision == "double")
          ExaDG::run<3, double>(input_file, degree, refine_space, refine_time, mpi_comm);
        else
          AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

  return 0;
}
