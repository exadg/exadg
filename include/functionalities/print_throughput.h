/*
 * print_throughput.h
 *
 *  Created on: Jun 11, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_PRINT_THROUGHPUT_H_
#define INCLUDE_FUNCTIONALITIES_PRINT_THROUGHPUT_H_


void
print_throughput(std::vector<std::pair<unsigned int, double>> const & wall_times,
                 std::string const &                                  name)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    // clang-format off
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Operator type: " << name
              << std::endl << std::endl
              << std::setw(5) << std::left << "k"
              << std::setw(15) << std::left << "DoFs/sec"
              << std::setw(15) << std::left << "DoFs/(sec*core)" << std::endl;

    typedef typename std::vector<std::pair<unsigned int, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::setw(5) << std::left << it->first
                << std::scientific << std::setprecision(4)
                << std::setw(15) << std::left << it->second
                << std::setw(15) << std::left << it->second/(double)N_mpi_processes
                << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
    // clang-format on
  }
}

void
print_throughput(
  std::vector<std::tuple<unsigned int, types::global_dof_index, double>> const & wall_times,
  std::string const &                                                            name)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    // clang-format off
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Operator type: " << name
              << std::endl << std::endl
              << std::setw(5) << std::left << "k"
              << std::setw(15) << std::left << "DoFs"
              << std::setw(15) << std::left << "DoFs/sec"
              << std::setw(15) << std::left << "DoFs/(sec*core)" << std::endl;

    typedef typename std::vector<std::tuple<unsigned int, types::global_dof_index, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::setw(5) << std::left << std::get<0>(*it)
                << std::scientific << std::setprecision(4)
                << std::setw(15) << std::left << (double)std::get<1>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)/(double)N_mpi_processes
                << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
    // clang-format on
  }
}


#endif /* INCLUDE_FUNCTIONALITIES_PRINT_THROUGHPUT_H_ */
