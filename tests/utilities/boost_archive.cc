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
  setup();
  void
  write_and_read();
  void
  check();

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;
  IndexSet           locally_owned_dofs;

  LinearAlgebra::distributed::Vector<double> vector_out;
  LinearAlgebra::distributed::Vector<double> vector_in;
};

template<int dim>
ArchiveVector<dim>::ArchiveVector()
  : mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
}

template<int dim>
void
ArchiveVector<dim>::setup()
{
  pcout << "Setting up vector.\n";

  parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(4);

  DoFHandler<dim> dof_handler(triangulation);
  const FE_Q<dim> fe(2);
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs = dof_handler.locally_owned_dofs();

  // Fill vector with global indices
  pcout << "Filling vector with ordered global indices [0, " << locally_owned_dofs.size() << ").\n";

  vector_out.reinit(locally_owned_dofs, mpi_communicator);

  double const this_mpi_process =
    static_cast<double>(Utilities::MPI::this_mpi_process(mpi_communicator));

  for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
  {
    vector_out.local_element(i) = static_cast<double>(locally_owned_dofs.nth_index_in_set(i));
  }
}

template<int dim>
void
ArchiveVector<dim>::write_and_read()
{
  std::string filename =
    "vector_proc_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator));

  pcout << "Storing the vector in the archive.\n";
  {
    std::ofstream stream(filename);
    AssertThrow(stream, ExcMessage("Could not write to file."));

    boost::archive::text_oarchive output_archive(stream);

    ExaDG::write_distributed_vector(vector_out, output_archive);
  }

  vector_in.reinit(locally_owned_dofs, mpi_communicator);

  pcout << "Reading the vector from the archive.\n";
  {
    std::ifstream stream(filename);
    AssertThrow(stream, ExcMessage("Could not read from file."));

    boost::archive::text_iarchive input_archive(stream);

    ExaDG::read_distributed_vector(vector_in, input_archive);
  }
}

template<int dim>
void
ArchiveVector<dim>::check()
{
  double norm = vector_out.linfty_norm();
  pcout << "vector_out.linfty_norm() = " << norm << "\n";
  norm = vector_out.l2_norm();
  pcout << "vector_out.l2_norm()     = " << norm << "\n\n";

  vector_in -= vector_out;

  norm = vector_in.linfty_norm();
  pcout << "error in linfty_norm = " << norm << "\n";
  norm = vector_in.l2_norm();
  pcout << "error in l2_norm     = " << norm << "\n";
}

template<int dim>
void
ArchiveVector<dim>::run()
{
  setup();
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
