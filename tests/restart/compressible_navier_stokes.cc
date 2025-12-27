/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/compressible_navier_stokes/driver.h>
#include <exadg/compressible_navier_stokes/user_interface/declare_get_application.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/time_integration/resolution_parameters.h>
#include <exadg/utilities/general_parameters.h>

namespace ExaDG
{
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

  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
    CompNS::get_application<dim, Number>(input_file, mpi_comm);

  application->set_parameters_convergence_study(degree, refine_space, refine_time);

  std::shared_ptr<CompNS::Driver<dim, Number>> driver =
    std::make_shared<CompNS::Driver<dim, Number>>(mpi_comm, application, is_test, false);

  driver->setup();

  driver->solve();

  if(not(is_test))
    driver->print_performance_results(timer.wall_time());
}

// Assert some parameter settings that should not be tested.
void
assert_non_test_parameters(ExaDG::GeneralParameters const &                 general,
                           ExaDG::SpatialResolutionParametersMinMax const & spatial,
                           ExaDG::TemporalResolutionParameters const &      temporal)
{
  AssertThrow(spatial.degree_min == spatial.degree_max,
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
  AssertThrow(general.precision == "double",
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
  AssertThrow(general.is_test == true,
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
  AssertThrow(general.dim == 2,
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
  AssertThrow(temporal.refine_time_min == temporal.refine_time_max,
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
  AssertThrow(spatial.refine_space_min == spatial.refine_space_max,
              dealii::ExcMessage("Invalid parameter combination for this test, check input file."));
}

} // namespace ExaDG

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // split default communicator: reference run in serial to store coarse grid
    int const rank  = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    int const color = rank == 0 ? 0 : MPI_UNDEFINED;
    MPI_Comm  mpi_comm_serial;
    MPI_Comm_split(mpi_comm, color, rank, &mpi_comm_serial);

    std::string const json_folder =
      "../../../../../tests/restart/json_files_compressible_navier_stokes/";

    // define *pairs of* reference and restart files
    std::vector<std::string> const reference_files = {"input_reference_read_write_mapping_N.json",
                                                      "input_reference_read_write_mapping_Y.json"};
    std::vector<std::string> const restart_files   = {"input_restart_read_write_mapping_N.json",
                                                    "input_restart_read_write_mapping_Y.json"};

    AssertThrow(reference_files.size() == restart_files.size(),
                dealii::ExcMessage(
                  "Equal number of `reference_files` and `restart_files` required."));

    for(unsigned int i = 0; i < reference_files.size(); ++i)
    {
      // serial reference run
      std::string const input_file_reference = json_folder + reference_files[i];

      ExaDG::GeneralParameters                 general_reference(input_file_reference);
      ExaDG::SpatialResolutionParametersMinMax spatial_reference(input_file_reference);
      ExaDG::TemporalResolutionParameters      temporal_reference(input_file_reference);

      // only allow select parameter combinations
      ExaDG::assert_non_test_parameters(general_reference, spatial_reference, temporal_reference);

      if(rank == 0)
      {
        std::cout << "\n\n\n--> REFERENCE input file : " << reference_files[i] << "\n";
        ExaDG::run<2, double>(input_file_reference,
                              spatial_reference.degree_min,
                              spatial_reference.refine_space_min,
                              temporal_reference.refine_time_min,
                              mpi_comm_serial /* SERIAL */,
                              general_reference.is_test);
      }
      MPI_Barrier(mpi_comm);

      // *parallel* restarted run
      std::string const                        input_file = json_folder + restart_files[i];
      ExaDG::GeneralParameters                 general(input_file);
      ExaDG::SpatialResolutionParametersMinMax spatial(input_file);
      ExaDG::TemporalResolutionParameters      temporal(input_file);

      // only allow select parameter combinations
      ExaDG::assert_non_test_parameters(general, spatial, temporal);

      if(rank == 0)
      {
        std::cout << "\n\n\n--> RESTART input file : " << restart_files[i] << "\n";
      }
      ExaDG::run<2, double>(input_file,
                            spatial.degree_min,
                            spatial.refine_space_min,
                            temporal.refine_time_min,
                            mpi_comm /* STANDARD */,
                            general.is_test);
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}

// application
#include "../applications/compressible_navier_stokes/manufactured_solution/application.h"
