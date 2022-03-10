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

#ifndef INCLUDE_EXADG_POISSON_OVERSET_GRIDS_SOLVER_H_
#define INCLUDE_EXADG_POISSON_OVERSET_GRIDS_SOLVER_H_

// ExaDG

// driver
#include <exadg/poisson/overset_grids/driver.h>

// utilities
#include <exadg/utilities/general_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

// application
#include <exadg/poisson/overset_grids/user_interface/declare_get_application.h>

namespace ExaDG
{
void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;
  Poisson::get_application_overset_grids<Dim, 1, Number>(input_file, MPI_COMM_WORLD)
    ->add_parameters(prm);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, int n_components, typename Number>
void
run(std::string const & input_file, MPI_Comm const & mpi_comm)
{
  std::shared_ptr<Poisson::ApplicationOversetGridsBase<dim, n_components, Number>> application =
    Poisson::get_application_overset_grids<dim, n_components, Number>(input_file, mpi_comm);

  std::shared_ptr<Poisson::DriverOversetGrids<dim, n_components, Number>> driver =
    std::make_shared<Poisson::DriverOversetGrids<dim, n_components, Number>>(mpi_comm, application);

  driver->setup();

  driver->solve();
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

  ExaDG::GeneralParameters general(input_file);

  if(general.dim == 2 && general.precision == "float")
    ExaDG::run<2, 1, float>(input_file, mpi_comm);
  else if(general.dim == 2 && general.precision == "double")
    ExaDG::run<2, 1, double>(input_file, mpi_comm);
  else if(general.dim == 3 && general.precision == "float")
    ExaDG::run<3, 1, float>(input_file, mpi_comm);
  else if(general.dim == 3 && general.precision == "double")
    ExaDG::run<3, 1, double>(input_file, mpi_comm);
  else
    AssertThrow(false,
                dealii::ExcMessage("Only dim = 2|3 and precision = float|double implemented."));

  return 0;
}


#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_SOLVER_H_ */
