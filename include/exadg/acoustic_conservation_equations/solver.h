/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SOLVER_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SOLVER_H_

// driver
#include <exadg/acoustic_conservation_equations/driver.h>

// utilities
#include <exadg/operators/resolution_parameters.h>
#include <exadg/time_integration/resolution_parameters.h>
#include <exadg/utilities/general_parameters.h>

// application
#include <exadg/acoustic_conservation_equations/user_interface/declare_get_application.h>

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

  TemporalResolutionParameters temporal;
  temporal.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  using Number           = double;
  Acoustics::get_application<Dim, Number>(input_file, MPI_COMM_WORLD)->add_parameters(prm);

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
  dealii::Timer timer;
  timer.restart();

  std::shared_ptr<Acoustics::ApplicationBase<dim, Number>> application =
    Acoustics::get_application<dim, Number>(input_file, mpi_comm);

  application->set_parameters_convergence_study(degree, refine_space, refine_time);

  std::shared_ptr<Acoustics::Driver<dim, Number>> driver =
    std::make_shared<Acoustics::Driver<dim, Number>>(mpi_comm, application, is_test, false);

  driver->setup();

  driver->solve();

  if(not(is_test))
    driver->print_performance_results(timer.wall_time());
}

} // namespace ExaDG

//#define USE_SUB_COMMUNICATOR

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  // new communicator
  MPI_Comm sub_comm;

#ifdef USE_SUB_COMMUNICATOR
  // use stride of n cores
  int const n = 48; // 24;

  int const rank     = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
  int const size     = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
  int const flag     = 1;
  int const new_rank = rank + (rank % n) * size;

  // split default communicator into two groups
  MPI_Comm_split(mpi_comm, flag, new_rank, &sub_comm);

  if(rank == 0)
    std::cout << std::endl << "Created sub communicator with stride of " << n << std::endl;
#else
  sub_comm = mpi_comm;
#endif

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

  ExaDG::GeneralParameters                 general(input_file);
  ExaDG::SpatialResolutionParametersMinMax spatial(input_file);
  ExaDG::TemporalResolutionParameters      temporal(input_file);

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
        if(general.dim == 2 and general.precision == "float")
        {
          ExaDG::run<2, float>(
            input_file, degree, refine_space, refine_time, sub_comm, general.is_test);
        }
        else if(general.dim == 2 and general.precision == "double")
        {
          ExaDG::run<2, double>(
            input_file, degree, refine_space, refine_time, sub_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "float")
        {
          ExaDG::run<3, float>(
            input_file, degree, refine_space, refine_time, sub_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "double")
        {
          ExaDG::run<3, double>(
            input_file, degree, refine_space, refine_time, sub_comm, general.is_test);
        }
        else
        {
          AssertThrow(
            false, dealii::ExcMessage("Only dim = 2|3 and precision = float|double implemented."));
        }
      }
    }
  }

#ifdef USE_SUB_COMMUNICATOR
  // free communicator
  MPI_Comm_free(&sub_comm);
#endif

  return 0;
}

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SOLVER_H_ */
