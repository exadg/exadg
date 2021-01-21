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
#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/utilities/general_parameters.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<FSI::ApplicationBase<dim, Number>>
get_application(std::string input_file);

template<int dim, typename Number>
void
add_parameters_application(dealii::ParameterHandler & prm, std::string const & input_file);

struct ResolutionParameters
{
  ResolutionParameters()
  {
  }

  ResolutionParameters(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("SpatialResolution");
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

  unsigned int degree_fluid = 3, degree_structure = 3;

  unsigned int refine_fluid = 0, refine_structure = 0;
};

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  ResolutionParameters resolution;
  resolution.add_parameters(prm);

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
run(std::string const &          input_file,
    ResolutionParameters const & resolution,
    MPI_Comm const &             mpi_comm,
    bool const                   is_test)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<FSI::Driver<dim, Number>> driver;
  driver.reset(new FSI::Driver<dim, Number>(input_file, mpi_comm));

  std::shared_ptr<FSI::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

  driver->setup(application,
                resolution.degree_fluid,
                resolution.degree_structure,
                resolution.refine_fluid,
                resolution.refine_structure,
                is_test);

  driver->solve();

  driver->print_performance_results(timer.wall_time(), is_test);
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1)
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To setup the input file, use: ./solver input_file --help" << std::endl;
      // clang-format on
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[1]);

    if(argc == 3 and std::string(argv[2]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        ExaDG::create_input_file(input_file);

      return 0;
    }
  }

  ExaDG::GeneralParameters    general(input_file);
  ExaDG::ResolutionParameters resolution(input_file);

  // run the simulation
  if(general.dim == 2 && general.precision == "float")
    ExaDG::run<2, float>(input_file, resolution, mpi_comm, general.is_test);
  else if(general.dim == 2 && general.precision == "double")
    ExaDG::run<2, double>(input_file, resolution, mpi_comm, general.is_test);
  else if(general.dim == 3 && general.precision == "float")
    ExaDG::run<3, float>(input_file, resolution, mpi_comm, general.is_test);
  else if(general.dim == 3 && general.precision == "double")
    ExaDG::run<3, double>(input_file, resolution, mpi_comm, general.is_test);
  else
    AssertThrow(false,
                dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));

  return 0;
}

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SOLVER_H_ */
