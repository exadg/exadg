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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_THROUGHPUT_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_THROUGHPUT_H_

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// driver
#include <exadg/compressible_navier_stokes/driver.h>

// utilities
#include <exadg/utilities/general_parameters.h>
#include <exadg/utilities/hypercube_resolution_parameters.h>
#include <exadg/utilities/throughput_parameters.h>

// application
#include <exadg/compressible_navier_stokes/user_interface/declare_get_application.h>

namespace ExaDG
{
void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  HypercubeResolutionParameters resolution;
  resolution.add_parameters(prm);

  ThroughputParameters throughput;
  throughput.add_parameters(prm);

  try
  {
    // we have to assume a default dimension and default Number type
    // for the automatic generation of a default input file
    unsigned int const Dim = 2;
    typedef double     Number;
    CompNS::get_application<Dim, Number>(input_file, MPI_COMM_WORLD)->add_parameters(prm);
  }
  catch(...)
  {
  }

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}


template<int dim, typename Number>
void
run(ThroughputParameters const & throughput,
    std::string const &          input_file,
    unsigned int const           degree,
    unsigned int const           refine_space,
    unsigned int const           n_cells_1d,
    MPI_Comm const &             mpi_comm,
    bool const                   is_test)
{
  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
    CompNS::get_application<dim, Number>(input_file, mpi_comm);

  application->set_parameters_throughput_study(degree, refine_space, n_cells_1d);

  std::shared_ptr<CompNS::Driver<dim, Number>> driver =
    std::make_shared<CompNS::Driver<dim, Number>>(mpi_comm, is_test, true);

  driver->setup(application);

  std::tuple<unsigned int, types::global_dof_index, double> wall_time =
    driver->apply_operator(throughput.operator_type,
                           throughput.n_repetitions_inner,
                           throughput.n_repetitions_outer);

  throughput.wall_times.push_back(wall_time);
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1)
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::cout << "To run the program, use:      ./throughput input_file" << std::endl
                << "To setup the input file, use: ./throughput input_file --help" << std::endl;
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

  ExaDG::GeneralParameters             general(input_file);
  ExaDG::HypercubeResolutionParameters resolution(input_file, general.dim);
  ExaDG::ThroughputParameters          throughput(input_file);

  // fill resolution vector depending on the operator_type
  resolution.fill_resolution_vector(&ExaDG::CompNS::get_dofs_per_element, input_file);

  // loop over resolutions vector and run simulations
  for(auto iter = resolution.resolutions.begin(); iter != resolution.resolutions.end(); ++iter)
  {
    unsigned int const degree       = std::get<0>(*iter);
    unsigned int const refine_space = std::get<1>(*iter);
    unsigned int const n_cells_1d   = std::get<2>(*iter);

    if(general.dim == 2 && general.precision == "float")
      ExaDG::run<2, float>(
        throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 2 && general.precision == "double")
      ExaDG::run<2, double>(
        throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 3 && general.precision == "float")
      ExaDG::run<3, float>(
        throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else if(general.dim == 3 && general.precision == "double")
      ExaDG::run<3, double>(
        throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm, general.is_test);
    else
      AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
  }

  if(not(general.is_test))
    throughput.print_results(mpi_comm);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_THROUGHPUT_H_ */
