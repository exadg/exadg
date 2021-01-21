/*
 * solver.h
 *
 *  Created on: Jan 19, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_SOLVER_H_
#define INCLUDE_EXADG_STRUCTURE_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// ExaDG

// driver
#include <exadg/structure/driver.h>

// utilities
#include <exadg/utilities/general_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<Structure::ApplicationBase<dim, Number>>
get_application(std::string input_file);

template<int dim, typename Number>
void
add_parameters_application(dealii::ParameterHandler & prm, std::string const & input_file);

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  SpatialResolutionParameters spatial;
  spatial.add_parameters(prm);

  TemporalResolutionParameters temporal;
  temporal.add_parameters(prm);

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
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm,
    bool const          is_test)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<Structure::Driver<dim, Number>> driver;
  driver.reset(new Structure::Driver<dim, Number>(mpi_comm));

  std::shared_ptr<Structure::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

  driver->setup(application, degree, refine_space, refine_time, is_test, false);

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

  ExaDG::GeneralParameters            general(input_file);
  ExaDG::SpatialResolutionParameters  spatial(input_file);
  ExaDG::TemporalResolutionParameters temporal(input_file);

  // k-refinement
  for(unsigned int degree = spatial.degree_min; degree <= spatial.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = spatial.refine_space_min;
        refine_space <= spatial.refine_space_max;
        ++refine_space)
    {
      // dt-refinement
      for(unsigned int refine_time = temporal.refine_time_min;
          refine_time <= temporal.refine_time_max;
          ++refine_time)
      {
        // run the simulation
        if(general.dim == 2 && general.precision == "float")
          ExaDG::run<2, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 2 && general.precision == "double")
          ExaDG::run<2, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 3 && general.precision == "float")
          ExaDG::run<3, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 3 && general.precision == "double")
          ExaDG::run<3, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else
          AssertThrow(false,
                      dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

  return 0;
}

#endif /* INCLUDE_EXADG_STRUCTURE_SOLVER_H_ */
