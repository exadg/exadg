
// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/revision.h>

// elasticity solver
#include "../include/structure/driver.h"
#include "./structure/bar/bar.h"
#include "./structure/beam/beam.h"
#include "./structure/can/can.h"

class ApplicationSelector
{
public:
  template<int dim, typename Number>
  void
  add_parameters(ParameterHandler & prm, std::string name_of_application = "")
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

      std::shared_ptr<Structure::ApplicationBase<dim, Number>> app;
      if(name == "Bar")
        app.reset(new Structure::Bar::Application<dim, Number>());
      else if(name == "Beam")
        app.reset(new Structure::Beam::Application<dim, Number>());
      else if(name == "Can")
        app.reset(new Structure::Can::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<Structure::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm);

    std::shared_ptr<Structure::ApplicationBase<dim, Number>> app;
    if(name == "Bar")
      app.reset(new Structure::Bar::Application<dim, Number>(input_file));
    else if(name == "Beam")
      app.reset(new Structure::Beam::Application<dim, Number>(input_file));
    else if(name == "Can")
      app.reset(new Structure::Can::Application<dim, Number>(input_file));
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

template<int dim, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    MPI_Comm const &    mpi_comm)
{
  std::shared_ptr<Structure::Driver<dim, Number>> solver;
  solver.reset(new Structure::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<Structure::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space);

  solver->solve();

  solver->analyze_computing_times();
}

struct ConvergenceStudy
{
  ConvergenceStudy()
  {
  }

  ConvergenceStudy(const std::string & input_file)
  {
    ParameterHandler prm;
    this->add_parameters(prm);

    parse_input(input_file, prm);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",      precision,        "Floating point precision.",                     Patterns::Selection("float|double"));
      prm.add_parameter("Dim",            dim,              "Number of space dimension.",                    Patterns::Integer(2,3));
      prm.add_parameter("DegreeMin",      degree_min,       "Minimal polynomial degree of shape functions.", Patterns::Integer(1,15));
      prm.add_parameter("DegreeMax",      degree_max,       "Maximal polynomial degree of shape functions.", Patterns::Integer(1,15));
      prm.add_parameter("RefineSpaceMin", refine_space_min, "Minimal number of mesh refinements.",           Patterns::Integer(0,20));
      prm.add_parameter("RefineSpaceMax", refine_space_max, "Maximal number of mesh refinements.",           Patterns::Integer(0,20));
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree_min = 3;
  unsigned int degree_max = 3;

  unsigned int refine_space_min = 0;
  unsigned int refine_space_max = 0;
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

using Number = double;

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // check if parameter file is provided

    // ./elasticity
    AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

    // ./elasticity --help
    if(argc == 2 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file();

      return 0;
    }
    // ./elasticity --help NameOfApplication
    else if(argc == 3 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file(argv[2]);

      return 0;
    }

    // the second argument is the input-file
    // ./elasticity InputFile
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
        // run the simulation
        if(study.dim == 2 && study.precision == "float")
          run<2, float>(input_file, degree, refine_space, mpi_comm);
        else if(study.dim == 2 && study.precision == "double")
          run<2, double>(input_file, degree, refine_space, mpi_comm);
        else if(study.dim == 3 && study.precision == "float")
          run<3, float>(input_file, degree, refine_space, mpi_comm);
        else if(study.dim == 3 && study.precision == "double")
          run<3, double>(input_file, degree, refine_space, mpi_comm);
        else
          AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
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
