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

#include <deal.II/base/revision.h>

#include <exadg/utilities/exadg_revision.h>
#include <exadg/utilities/print_general_infos.h>


namespace ExaDG
{
void
print_exadg_header(dealii::ConditionalOStream const & pcout)
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
        << print_horizontal_line() << std::endl
        << "                                                                                " << std::endl
        << "                ////////                      ///////   ////////                " << std::endl
        << "                ///                           ///  ///  ///                     " << std::endl
        << "                //////    ///  ///  ///////   ///  ///  /// ////                " << std::endl
        << "                ///         ////    //   //   ///  ///  ///  ///                " << std::endl
        << "                ////////  ///  ///  ///////// ///////   ////////                " << std::endl
        << "                                                                                " << std::endl
        << "               High-Order Discontinuous Galerkin for the Exa-Scale              " << std::endl
        << print_horizontal_line() << std::endl << std::endl;
  // clang-format on
}

// print MPI info
void
print_MPI_info(dealii::ConditionalOStream const & pcout, MPI_Comm const & mpi_comm)
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout, "Number of processes", dealii::Utilities::MPI::n_mpi_processes(mpi_comm));
}



// print deal.II info
void
print_dealii_info(dealii::ConditionalOStream const & pcout)
{
  pcout << std::endl
        << "deal.II info:" << std::endl
        << std::endl
        << "  deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH
        << std::endl;
}

// print ExaDG info
void
print_exadg_info(dealii::ConditionalOStream const & pcout)
{
  pcout << std::endl
        << "ExaDG info:" << std::endl
        << std::endl
        << "  ExaDG git version " << EXADG_GIT_SHORTREV << " on branch " << EXADG_GIT_BRANCH
        << std::endl;
}

} // namespace ExaDG
