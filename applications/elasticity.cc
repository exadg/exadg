
// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// elasticity solver
#include "../include/structure/driver.h"

// infrastructure for convergence studies
#include "../include/utilities/convergence_study.h"

// applications
#include "./structure/bar/bar.h"
#include "./structure/beam/beam.h"
#include "./structure/can/can.h"
#include "./structure/manufactured/manufactured.h"

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

      std::shared_ptr<Structure::ApplicationBase<dim, Number>> app;
      if(name == "Bar")
        app.reset(new Structure::Bar::Application<dim, Number>());
      else if(name == "Beam")
        app.reset(new Structure::Beam::Application<dim, Number>());
      else if(name == "Can")
        app.reset(new Structure::Can::Application<dim, Number>());
      else if(name == "Manufactured")
        app.reset(new Structure::Manufactured::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<Structure::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

    std::shared_ptr<Structure::ApplicationBase<dim, Number>> app;
    if(name == "Bar")
      app.reset(new Structure::Bar::Application<dim, Number>(input_file));
    else if(name == "Beam")
      app.reset(new Structure::Beam::Application<dim, Number>(input_file));
    else if(name == "Can")
      app.reset(new Structure::Can::Application<dim, Number>(input_file));
    else if(name == "Manufactured")
      app.reset(new Structure::Manufactured::Application<dim, Number>(input_file));
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
  Timer timer;
  timer.restart();

  std::shared_ptr<Structure::Driver<dim, Number>> solver;
  solver.reset(new Structure::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<Structure::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space, refine_time);

  solver->solve();

  solver->print_statistics(timer.wall_time());
}

int
main(int argc, char ** argv)
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
    for(unsigned int refine_space = study.refine_space_min; refine_space <= study.refine_space_max;
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
          AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

  return 0;
}
