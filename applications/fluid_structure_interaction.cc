/*
 * fluid_structure_interaction.cc
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/parameter_handler.h>

// driver
#include "../include/fluid_structure_interaction/driver.h"

// application

// template
#include "fluid_structure_interaction_test_cases/template/template.h"

#include "fluid_structure_interaction_test_cases/bending_wall/bending_wall.h"
#include "fluid_structure_interaction_test_cases/cylinder_with_flag/cylinder_with_flag.h"
#include "fluid_structure_interaction_test_cases/pressure_wave/pressure_wave.h"

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
      prm.add_parameter("DegreeFluid",
                        degree_fluid,
                        "Polynomial degree of fluid (velocity).",
                        Patterns::Integer(1,15),
                        true);
      prm.add_parameter("DegreeStructure",
                        degree_structure,
                        "Polynomial degree of structural problem.",
                        Patterns::Integer(1,15),
                        true);
      prm.add_parameter("RefineFluid",
                        refine_fluid,
                        "Number of mesh refinements (fluid).",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineStructure",
                        refine_structure,
                        "Number of mesh refinements (structure).",
                        Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree_fluid = 3, degree_structure = 3;

  unsigned int refine_fluid = 0, refine_structure = 0;
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

  FSI::PartitionedData fsi_data;
  FSI::Driver<Dim, Number>::add_parameters(prm, fsi_data);

  ApplicationSelector selector;
  selector.add_parameters<Dim, Number>(prm, input_file);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(std::string const & input_file, Study const & study, MPI_Comm const & mpi_comm)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<FSI::Driver<dim, Number>> driver;
  driver.reset(new FSI::Driver<dim, Number>(input_file, mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<FSI::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  driver->setup(application,
                study.degree_fluid,
                study.degree_structure,
                study.refine_fluid,
                study.refine_structure);

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
    ExaDG::run<2, float>(input_file, study, mpi_comm);
  else if(study.dim == 2 && study.precision == "double")
    ExaDG::run<2, double>(input_file, study, mpi_comm);
  else if(study.dim == 3 && study.precision == "float")
    ExaDG::run<3, float>(input_file, study, mpi_comm);
  else if(study.dim == 3 && study.precision == "double")
    ExaDG::run<3, double>(input_file, study, mpi_comm);
  else
    AssertThrow(false,
                dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));

  return 0;
}
