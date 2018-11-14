/*
 * restart.h
 *
 *  Created on: Nov 13, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_RESTART_H_
#define INCLUDE_TIME_INTEGRATION_RESTART_H_


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <fstream>
#include <sstream>

inline std::string
restart_filename(std::string const & name)
{
  std::string const rank =
    Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  std::string const filename = name + "." + rank + ".restart";

  return filename;
}

inline void
read_restart_preamble(boost::archive::binary_iarchive & ia,
                      double &                          time,
                      std::vector<double> &             time_steps,
                      unsigned int const                order)
{
  unsigned int n_old_ranks = 1;
  unsigned int old_order   = 1;

  // Note that the operations done here must be in sync with the output.
  ia & n_old_ranks;
  ia & time;
  ia & old_order;

  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  AssertThrow(n_old_ranks == n_ranks,
              ExcMessage("Tried to restart with " + Utilities::to_string(n_ranks) +
                         " processes, "
                         "but restart was written on " +
                         Utilities::to_string(n_old_ranks) + " processes."));

  AssertThrow(old_order == order, ExcMessage("Order of time integrator may not change."));

  for(unsigned int i = 0; i < order; i++)
    ia & time_steps[i];
}

inline void
write_restart_preamble(boost::archive::binary_oarchive & oa,
                       std::string const &               name,
                       double const                      time,
                       std::vector<double> const &       time_steps,
                       unsigned int const                order)
{
  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  oa & n_ranks;
  oa & time;
  oa & order;

  for(unsigned int i = 0; i < order; i++)
    oa & time_steps[i];

  // backup: rename current restart file into restart.old in case something fails while writing
  std::string const rank =
    Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
  std::string const from = restart_filename(name);
  std::string const to   = restart_filename(name) + ".old";

  std::ifstream ifile(from.c_str());
  if((bool)ifile) // rename only if file already exists
  {
    int const error = rename(from.c_str(), to.c_str());

    AssertThrow(error == 0, ExcMessage("Can not rename file: " + from + " -> " + to));
  }
}

inline void
write_restart_file(std::ostringstream & oss, std::string const & name)
{
  const std::string filename = restart_filename(name);
  std::ofstream     stream(filename.c_str());

  stream << oss.str() << std::endl;
}



#endif /* INCLUDE_TIME_INTEGRATION_RESTART_H_ */
