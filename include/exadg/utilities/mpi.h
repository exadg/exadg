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

#ifndef INCLUDE_UTILITIES_MPI_H_
#define INCLUDE_UTILITIES_MPI_H_

namespace ExaDG
{
/*
 * Return whether the current MPI process is the first on a compute node,
 * defined as the ranks which share the same memory
 * (`MPI_COMM_TYPE_SHARED`). The second argument returns the number of
 * processes per node, in case it is needed in the algorithm.
 */
std::tuple<bool, unsigned int>
identify_first_process_on_node(MPI_Comm const & mpi_comm)
{
  int rank;
  MPI_Comm_rank(mpi_comm, &rank);

  MPI_Comm comm_shared;
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_shared);

  int size_shared;
  MPI_Comm_size(comm_shared, &size_shared);
  MPI_Comm_free(&comm_shared);

  AssertThrow(size_shared == dealii::Utilities::MPI::max(size_shared, mpi_comm),
              dealii::ExcMessage(
                "The identification of MPI process groups in terms of compute nodes only "
                "works if all nodes are populated with the same number of MPI ranks!"));
  return {rank % size_shared == 0, size_shared};
}

} // namespace ExaDG

#endif
