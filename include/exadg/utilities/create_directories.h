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

#ifndef INCLUDE_EXADG_UTILITIES_CREATE_DIRECTORIES_H_
#define INCLUDE_EXADG_UTILITIES_CREATE_DIRECTORIES_H_

// C/C++
#include <filesystem>

namespace ExaDG
{
/**
 * Creates directories if not already existing. An MPI barrier ensures that
 * directories have been created for all processes when completing this function.
 */
inline void
create_directories(std::string const & directory, MPI_Comm const & mpi_comm)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    std::filesystem::create_directories(directory);

  MPI_Barrier(mpi_comm);
}

} // namespace ExaDG



#endif /* INCLUDE_EXADG_UTILITIES_CREATE_DIRECTORIES_H_ */
