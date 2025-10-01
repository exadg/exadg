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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// C/C++
#include <filesystem>
#include <fstream>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

// boost
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// ExaDG
#include <exadg/time_integration/restart.h>

using namespace dealii;

template<int dim>
class ArchiveVector
{
public:
  ArchiveVector();
  void
  run();

private:
  void
  write_and_read();
  void
  check();

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  static double constexpr number_to_archive = 1.23456;
  double number_read_from_archive           = 0.0;
};

template<int dim>
ArchiveVector<dim>::ArchiveVector()
  : mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
}

template<int dim>
void
ArchiveVector<dim>::write_and_read()
{
  std::string filename =
    "number_proc_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator));

  pcout << "Storing the number in the archive.\n";
  {
    std::ofstream stream(filename);
    AssertThrow(stream, ExcMessage("Could not write to file."));

    boost::archive::text_oarchive output_archive(stream);

    output_archive & number_to_archive;
  }

  pcout << "Reading the number from the archive.\n";
  {
    std::ifstream stream(filename);
    AssertThrow(stream, ExcMessage("Could not read from file."));

    boost::archive::text_iarchive input_archive(stream);

    input_archive & number_read_from_archive;
  }
}

template<int dim>
void
ArchiveVector<dim>::check()
{
  // Read the number with all processes from the archive and compute the maximum relative
  // difference.
  double norm =
    std::abs(number_to_archive - number_read_from_archive) / std::abs(number_to_archive);

  norm = Utilities::MPI::max(norm, mpi_communicator);

  pcout << "Maximum relative difference in numbers written and read over all processors:\n"
        << norm << "\n";
}

template<int dim>
void
ArchiveVector<dim>::run()
{
  write_and_read();
  check();
}

int
main(int argc, char * argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ArchiveVector<3> archive_vector;
    archive_vector.run();
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
