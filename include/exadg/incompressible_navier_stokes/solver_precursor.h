/*
 * solver_precursor.h
 *
 *  Created on: Jan 19, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SOLVER_PRECURSOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SOLVER_PRECURSOR_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG

// driver
#include <exadg/incompressible_navier_stokes/driver_precursor.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>>
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
      prm.add_parameter("Degree",
                        degree,
                        "Polynomial degree of shape functions.",
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineSpace",
                        refine_space,
                        "Number of global, uniform mesh refinements.",
                        Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree = 3;

  unsigned int refine_space = 0;
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

  add_parameters_application<Dim, Number>(prm, input_file);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
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

  std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>> application =
    get_application<dim, Number>(input_file);

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
      std::cout << "To run the program, use:      ./solver_precursor input_file" << std::endl
                << "To create an input file, use: ./solver_precursor --create_input_file input_file"
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
    ExaDG::run<2, float>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 2 && study.precision == "double")
    ExaDG::run<2, double>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 3 && study.precision == "float")
    ExaDG::run<3, float>(input_file, study.degree, study.refine_space, mpi_comm);
  else if(study.dim == 3 && study.precision == "double")
    ExaDG::run<3, double>(input_file, study.degree, study.refine_space, mpi_comm);
  else
    AssertThrow(false,
                dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));

  return 0;
}

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SOLVER_PRECURSOR_H_ */
