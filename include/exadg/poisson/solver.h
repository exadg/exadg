/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_POISSON_SOLVER_H_
#define INCLUDE_EXADG_POISSON_SOLVER_H_

// ExaDG

// driver
#include <exadg/poisson/driver.h>

// utilities
#include <exadg/grid/grid_data.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/utilities/general_parameters.h>

// application
#include <exadg/poisson/user_interface/declare_get_application.h>

namespace ExaDG
{
void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  SpatialResolutionParametersMinMax spatial;
  spatial.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;
  Poisson::get_application<Dim, 1, Number>(input_file, MPI_COMM_WORLD)->add_parameters(prm);

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
    MPI_Comm const &            mpi_comm,
    bool const                  is_test)
{
  dealii::Timer timer;
  timer.restart();

  std::shared_ptr<Poisson::ApplicationBase<dim, 1, Number>> application =
    Poisson::get_application<dim, 1, Number>(input_file, mpi_comm);

  application->set_parameters_convergence_study(degree, refine_space);

  std::shared_ptr<Poisson::Driver<dim, Number>> driver =
    std::make_shared<Poisson::Driver<dim, Number>>(mpi_comm, application, is_test, false);

  driver->setup();

  driver->solve();

  SolverResult result = driver->print_performance_results(timer.wall_time());
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

    if(argc == 3 and std::string(argv[2]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        ExaDG::create_input_file(input_file);

      return 0;
    }
  }

  ExaDG::GeneralParameters                 general(input_file);
  ExaDG::SpatialResolutionParametersMinMax spatial(input_file);

  std::vector<ExaDG::SolverResult> results;

  // k-refinement
  for(unsigned int degree = spatial.degree_min; degree <= spatial.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = spatial.refine_space_min;
        refine_space <= spatial.refine_space_max;
        ++refine_space)
    {
      if(general.dim == 2 and general.precision == "float")
      {
        ExaDG::run<2, float>(results, input_file, degree, refine_space, mpi_comm, general.is_test);
      }
      else if(general.dim == 2 and general.precision == "double")
      {
        ExaDG::run<2, double>(results, input_file, degree, refine_space, mpi_comm, general.is_test);
      }
      else if(general.dim == 3 and general.precision == "float")
      {
        ExaDG::run<3, float>(results, input_file, degree, refine_space, mpi_comm, general.is_test);
      }
      else if(general.dim == 3 and general.precision == "double")
      {
        ExaDG::run<3, double>(results, input_file, degree, refine_space, mpi_comm, general.is_test);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Only dim = 2|3 and precision = float|double implemented."));
      }
    }
  }

  if(not(general.is_test))
    print_results(results, mpi_comm);

  return 0;
}


#endif /* INCLUDE_EXADG_POISSON_SOLVER_H_ */
