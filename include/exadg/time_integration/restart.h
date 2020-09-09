/*
 * restart.h
 *
 *  Created on: Nov 13, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_

// C/C++
#include <fstream>
#include <sstream>

namespace ExaDG
{
using namespace dealii;

inline std::string
restart_filename(std::string const & name, MPI_Comm const & mpi_comm)
{
  std::string const rank = Utilities::int_to_string(Utilities::MPI::this_mpi_process(mpi_comm));

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

    AssertThrow(error == 0, ExcMessage("Can not rename file: " + from + " -> " + to));
  }
}

inline void
write_restart_file(std::ostringstream & oss, std::string const & filename)
{
  std::ofstream stream(filename.c_str());

  stream << oss.str() << std::endl;
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_ */
