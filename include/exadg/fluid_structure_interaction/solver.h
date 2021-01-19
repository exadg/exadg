/*
 * solver.h
 *
 *  Created on: Jan 19, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SOLVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SOLVER_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG

// driver
#include <exadg/fluid_structure_interaction/driver.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<FSI::ApplicationBase<dim, Number>>
get_application(std::string input_file);

template<int dim, typename Number>
void
add_parameters_application(dealii::ParameterHandler & prm, std::string const & input_file);

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
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("DegreeStructure",
                        degree_structure,
                        "Polynomial degree of structural problem.",
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
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

  add_parameters_application<Dim, Number>(prm, input_file);

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

  std::shared_ptr<FSI::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

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

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SOLVER_H_ */
