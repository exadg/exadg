/*
 * solver.h
 *
 *  Created on: Jan 19, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POISSON_SOLVER_H_
#define INCLUDE_EXADG_POISSON_SOLVER_H_

// ExaDG

// driver
#include <exadg/poisson/driver.h>

// utilities
#include <exadg/utilities/general_parameters.h>
#include <exadg/utilities/hypercube_resolution_parameters.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<Poisson::ApplicationBase<dim, Number>>
get_application(std::string input_file);

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  HypercubeResolutionParameters resolution;
  resolution.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;
  get_application<Dim, Number>(input_file)->add_parameters(prm);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(std::vector<SolverResult> & results,
    std::string const &         input_file,
    unsigned int const          degree,
    unsigned int const          refine_space,
    unsigned int const          n_cells_1d,
    MPI_Comm const &            mpi_comm,
    bool const                  is_test)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<Poisson::Driver<dim, Number>> driver;
  driver.reset(new Poisson::Driver<dim, Number>(mpi_comm));

  std::shared_ptr<Poisson::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

  application->set_subdivisions_hypercube(n_cells_1d);

  driver->setup(application, degree, refine_space, is_test, false);

  driver->solve();

  SolverResult result = driver->print_performance_results(timer.wall_time(), is_test);
  results.push_back(result);
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
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To setup the input file, use: ./solver input_file --help" << std::endl;
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[1]);

    if(argc == 3 && std::string(argv[2]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        ExaDG::create_input_file(input_file);

      return 0;
    }
  }

  ExaDG::GeneralParameters             general(input_file);
  ExaDG::HypercubeResolutionParameters resolution(input_file, general.dim);

  // fill resolution vector
  resolution.fill_resolution_vector(&ExaDG::Poisson::get_dofs_per_element, input_file);

  std::vector<ExaDG::SolverResult> results;

  // loop over resolutions vector and run simulations
  for(auto iter = resolution.resolutions.begin(); iter != resolution.resolutions.end(); ++iter)
  {
    unsigned int const degree       = std::get<0>(*iter);
    unsigned int const refine_space = std::get<1>(*iter);
    unsigned int const n_cells_1d   = std::get<2>(*iter);

    if(general.dim == 2 && general.precision == "float")
      ExaDG::run<2, float>(
        results, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 2 && general.precision == "double")
      ExaDG::run<2, double>(
        results, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 3 && general.precision == "float")
      ExaDG::run<3, float>(
        results, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 3 && general.precision == "double")
      ExaDG::run<3, double>(
        results, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else
      AssertThrow(false,
                  dealii::ExcMessage("Only dim = 2|3 and precision = float|double implemented."));
  }

  if(not(general.is_test))
    print_results(results, mpi_comm);

  return 0;
}


#endif /* INCLUDE_EXADG_POISSON_SOLVER_H_ */
