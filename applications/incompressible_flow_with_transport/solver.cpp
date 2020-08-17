/*
 * incompressible_flow_with_transport.cc
 *
 *  Created on: Nov 6, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/parameter_handler.h>

// driver
#include "../include/incompressible_flow_with_transport/driver.h"

// application
#include "applications/template/template.h"

// passive scalar
#include "applications/cavity/cavity.h"
#include "applications/lung/lung.h"

// natural convection (active scalar)
#include "applications/cavity_natural_convection/cavity_natural_convection.h"
#include "applications/mantle_convection/mantle_convection.h"
#include "applications/rayleigh_benard/rayleigh_benard.h"
#include "applications/rising_bubble/rising_bubble.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<FTI::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<FTI::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new FTI::Template::Application<dim, Number>(input_file));
    else if(name == "Cavity")
      app.reset(new FTI::Cavity::Application<dim, Number>(input_file));
    else if(name == "Lung")
      app.reset(new FTI::Lung::Application<dim, Number>(input_file));
    else if(name == "CavityNaturalConvection")
      app.reset(new FTI::CavityNaturalConvection::Application<dim, Number>(input_file));
    else if(name == "RayleighBenard")
      app.reset(new FTI::RayleighBenard::Application<dim, Number>(input_file));
    else if(name == "RisingBubble")
      app.reset(new FTI::RisingBubble::Application<dim, Number>(input_file));
    else if(name == "MantleConvection")
      app.reset(new FTI::MantleConvection::Application<dim, Number>(input_file));
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
      std::shared_ptr<FTI::ApplicationBase<dim, Number>> app =
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

struct Study
{
  Study()
  {
  }

  Study(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",
                        precision,
                        "Floating point precision.",
                        Patterns::Selection("float|double"),
                        false);
      prm.add_parameter("Dim",
                        dim,
                        "Number of space dimension.",
                        Patterns::Integer(2,3),
                        true);
      prm.add_parameter("Degree",
                        degree,
                        "Polynomial degree of shape functions.",
                        Patterns::Integer(1,15),
                        true);
      prm.add_parameter("RefineSpace",
                        refine_space,
                        "Number of global, uniform mesh refinements.",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("NScalars",
                        n_scalars,
                        "Number of scalar fields.",
                        Patterns::Integer(1,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree = 3;

  unsigned int refine_space = 0;

  unsigned int n_scalars = 1;
};

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  Study study;
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
    unsigned int const  n_scalars,
    MPI_Comm const &    mpi_comm)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<FTI::Driver<dim, Number>> driver;
  driver.reset(new FTI::Driver<dim, Number>(mpi_comm, n_scalars));

  ApplicationSelector selector;

  std::shared_ptr<FTI::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  driver->setup(application, degree, refine_space);

  driver->solve();

  driver->print_statistics(timer.wall_time());
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

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

  ExaDG::Study study(input_file);

  // run the simulation
  if(study.dim == 2 && study.precision == "float")
    ExaDG::run<2, float>(input_file, study.degree, study.refine_space, study.n_scalars, mpi_comm);
  else if(study.dim == 2 && study.precision == "double")
    ExaDG::run<2, double>(input_file, study.degree, study.refine_space, study.n_scalars, mpi_comm);
  else if(study.dim == 3 && study.precision == "float")
    ExaDG::run<3, float>(input_file, study.degree, study.refine_space, study.n_scalars, mpi_comm);
  else if(study.dim == 3 && study.precision == "double")
    ExaDG::run<3, double>(input_file, study.degree, study.refine_space, study.n_scalars, mpi_comm);
  else
    AssertThrow(false,
                dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));

  return 0;
}
