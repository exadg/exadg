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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_

// C/C++
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
inline std::string
restart_filename(std::string const & name, MPI_Comm const & mpi_comm)
{
  std::string const rank =
    dealii::Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process(mpi_comm));

  std::string const filename = name + "." + rank + ".restart";

  return filename;
}

inline void
rename_restart_files(std::string const & filename)
{
  // backup: rename current restart file into restart.old in case something fails while writing
  std::string const from = filename;
  std::string const to   = filename + ".old";

  std::ifstream ifile(from.c_str());
  if((bool)ifile) // rename only if file already exists
  {
    int const error = rename(from.c_str(), to.c_str());

    AssertThrow(error == 0, dealii::ExcMessage("Can not rename file: " + from + " -> " + to));
  }
}

inline void
write_restart_file(std::ostringstream & oss, std::string const & filename)
{
  std::ofstream stream(filename.c_str());

  stream << oss.str() << std::endl;
}

template<typename VectorType>
inline void
print_vector_l2_norm(VectorType const & vector)
{
  MPI_Comm const & mpi_comm = vector.get_mpi_communicator();
  double const     l2_norm  = vector.l2_norm();
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << "    vector global l2 norm: " << std::scientific << std::setprecision(8)
              << std::setw(20) << l2_norm << "\n";
  }
}

/**
 * Utility function to read or write the local entries of a
 * dealii::LinearAlgebra::distributed::(Block)Vector
 * from/to a boost archive per block and entry.
 * Using the `&` operator, loading from or writing to the
 * archive is determined from the type.
 */
template<typename VectorType, typename BoostArchiveType>
inline void
read_write_distributed_vector(VectorType & vector, BoostArchiveType & archive)
{
  // Print vector norm here only *before* writing.
  if(std::is_same<BoostArchiveType, boost::archive::text_oarchive>::value or
     std::is_same<BoostArchiveType, boost::archive::binary_oarchive>::value)
  {
    print_vector_l2_norm(vector);
  }

  // Depending on VectorType, we have to loop over the blocks to
  // access the local entries via vector.local_element(i).
  using Number = typename VectorType::value_type;
  if constexpr(std::is_same<std::remove_cv_t<VectorType>,
                            dealii::LinearAlgebra::distributed::Vector<Number>>::value)
  {
    for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
    {
      archive & vector.local_element(i);
    }
  }
  else if constexpr(std::is_same<VectorType,
                                 dealii::LinearAlgebra::distributed::BlockVector<Number>>::value or
                    std::is_same<
                      VectorType,
                      dealii::LinearAlgebra::distributed::BlockVector<Number> const>::value)
  {
    for(unsigned int i = 0; i < vector.n_blocks(); ++i)
    {
      for(unsigned int i = 0; i < vector.block(i).locally_owned_size(); ++i)
      {
        archive & vector.block(i).local_element(i);
      }
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Reading into this VectorType not supported."));
  }

  // Print vector norm here only *after* reading.
  if(std::is_same<BoostArchiveType, boost::archive::text_iarchive>::value or
     std::is_same<BoostArchiveType, boost::archive::binary_iarchive>::value)
  {
    print_vector_l2_norm(vector);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_ */
