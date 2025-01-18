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

#ifndef INCLUDE_EXADG_UTILITIES_BOOST_ARCHIVE_H_
#define INCLUDE_EXADG_UTILITIES_BOOST_ARCHIVE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/**
 * Utility functions to read and write the local entries of a
 * dealii::LinearAlgebra::distributed::Vector
 * from/to a boost archive per entry.
 */
template<typename Number, typename BoostInputArchiveType>
inline void
read_distributed_vector(dealii::LinearAlgebra::distributed::Vector<Number> & vector,
                        BoostInputArchiveType &                              input_archive)
{
  for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
  {
    input_archive >> vector.local_element(i);
  }

#ifdef DEBUG
  MPI_Comm const &   mpi_comm = vector.get_mpi_communicator();
  unsigned int const rank     = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
  std::cout << "    reading vector on rank " << rank << "; global l2 norm: " << vector.l2_norm()
            << std::endl;
#endif
}

template<typename Number, typename BoostOutputArchiveType>
inline void
write_distributed_vector(dealii::LinearAlgebra::distributed::Vector<Number> const & vector,
                         BoostOutputArchiveType &                                   output_archive)
{
#ifdef DEBUG
  MPI_Comm const &   mpi_comm = vector.get_mpi_communicator();
  unsigned int const rank     = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
  std::cout << "    writing vector on rank " << rank << "; global l2 norm: " << vector.l2_norm()
            << std::endl;
#endif

  for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
  {
    output_archive << vector.local_element(i);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_BOOST_ARCHIVE_H_ */
