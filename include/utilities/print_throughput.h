/*
 * print_throughput.h
 *
 *  Created on: Jun 11, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_UTILITIES_PRINT_THROUGHPUT_H_
#define INCLUDE_UTILITIES_PRINT_THROUGHPUT_H_


inline void
print_throughput(std::vector<std::pair<unsigned int, double>> const & wall_times,
                 std::string const &                                  name,
                 MPI_Comm const &                                     mpi_comm)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // clang-format off
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Operator type: " << name
              << std::endl << std::endl
              << std::setw(5) << std::left << "k"
              << std::setw(15) << std::left << "DoFs/sec"
              << std::setw(15) << std::left << "DoFs/(sec*core)"
              << std::endl << std::flush;

    typedef typename std::vector<std::pair<unsigned int, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::setw(5) << std::left << it->first
                << std::scientific << std::setprecision(4)
                << std::setw(15) << std::left << it->second
                << std::setw(15) << std::left << it->second/(double)N_mpi_processes
                << std::endl << std::flush;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl << std::flush;
    // clang-format on
  }
}

inline void
print_throughput(
  std::vector<std::tuple<unsigned int, types::global_dof_index, double>> const & wall_times,
  std::string const &                                                            name,
  MPI_Comm const &                                                               mpi_comm)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
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
              << std::setw(15) << std::left << "DoFs/(sec*core)"
              << std::endl << std::flush;

    typedef typename std::vector<std::tuple<unsigned int, types::global_dof_index, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::setw(5) << std::left << std::get<0>(*it)
                << std::scientific << std::setprecision(4)
                << std::setw(15) << std::left << (double)std::get<1>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)/(double)N_mpi_processes
                << std::endl << std::flush;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl << std::flush;
    // clang-format on
  }
}

inline void
print_throughput_steady(ConditionalOStream const &    pcout,
                        types::global_dof_index const n_dofs,
                        double const                  overall_time_avg,
                        unsigned int const            N_mpi_processes)
{
  // clang-format off
  pcout << std::endl
        << "Throughput:" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Degrees of freedom      = " << n_dofs << std::endl
        << "  Wall time               = " << std::scientific << std::setprecision(2) << overall_time_avg << " s" << std::endl
        << "  Throughput              = " << std::scientific << std::setprecision(2) << n_dofs / (overall_time_avg * N_mpi_processes) << " DoFs/s/core" << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_throughput_10(ConditionalOStream const &    pcout,
                    types::global_dof_index const n_dofs,
                    double const                  t_10,
                    unsigned int const            N_mpi_processes)
{
  double const tau_10 = t_10 * (double)N_mpi_processes / n_dofs;

  // clang-format off
  pcout << "Throughput of linear solver (numbers based on n_10):" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Degrees of freedom      = " << n_dofs << std::endl
        << "  Wall time t_10          = " << std::scientific << std::setprecision(2) << t_10 << " s" << std::endl
        << "  tau_10                  = " << std::scientific << std::setprecision(2) << tau_10 << " s*core/DoF" << std::endl
        << "  Throughput E_10         = " << std::scientific << std::setprecision(2) << 1.0 / tau_10 << " DoF/s/core" << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_throughput_unsteady(ConditionalOStream const &    pcout,
                          types::global_dof_index const n_dofs,
                          double const                  overall_time_avg,
                          unsigned int const            N_time_steps,
                          unsigned int const            N_mpi_processes)
{
  double const time_per_timestep = overall_time_avg / (double)N_time_steps;

  // clang-format off
  pcout << std::endl
        << "Throughput per time step:" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Degrees of freedom      = " << n_dofs << std::endl
        << "  Wall time               = " << std::scientific << std::setprecision(2) << overall_time_avg << " s" << std::endl
        << "  Time steps              = " << std::left << N_time_steps << std::endl
        << "  Wall time per time step = " << std::scientific << std::setprecision(2) << time_per_timestep << " s" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Throughput              = " << std::scientific << std::setprecision(2) << n_dofs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl
        << std::flush;
  // clang-format on
}


inline void
print_costs(ConditionalOStream const & pcout,
            double const               overall_time_avg,
            unsigned int const         N_mpi_processes)

{
  // clang-format off
  pcout << std::endl
        << "Computational costs:" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Wall time               = " << std::scientific << std::setprecision(2) << overall_time_avg << " s" << std::endl
        << "  Computational costs     = " << std::scientific << std::setprecision(2) << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_solver_info_nonlinear(ConditionalOStream const & pcout,
                            unsigned int const         N_iter_nonlinear,
                            unsigned int const         N_iter_linear,
                            double const               wall_time)

{
  double const N_iter_linear_avg =
    (N_iter_nonlinear > 0) ? double(N_iter_linear) / double(N_iter_nonlinear) : N_iter_linear;

  // clang-format off
  pcout << std::endl
        << "  Newton iterations:      " << std::setw(12) << std::right << N_iter_nonlinear << std::endl
        << "  Linear iterations (avg):" << std::setw(12) << std::fixed << std::setprecision(1) << std::right << N_iter_linear_avg << std::endl
        << "  Linear iterations (tot):" << std::setw(12) << std::right << N_iter_linear << std::endl
        << "  Wall time [s]:          " << std::setw(12) << std::scientific << std::setprecision(2) << std::right << wall_time << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_solver_info_linear(ConditionalOStream const & pcout,
                         unsigned int const         N_iter_linear,
                         double const               wall_time)

{
  // clang-format off
  pcout << std::endl
        << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
        << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(2) << std::right << wall_time << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_solver_info_explicit(ConditionalOStream const & pcout, double const wall_time)

{
  // clang-format off
  pcout << std::endl
        << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(2) << std::right << wall_time << std::endl
        << std::flush;
  // clang-format on
}


#endif /* INCLUDE_UTILITIES_PRINT_THROUGHPUT_H_ */
