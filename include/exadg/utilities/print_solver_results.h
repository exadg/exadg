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

#ifndef INCLUDE_EXADG_UTILITIES_PRINT_SOLVER_RESULTS_H_
#define INCLUDE_EXADG_UTILITIES_PRINT_SOLVER_RESULTS_H_

// C/C++
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
inline void
print_throughput(
  std::vector<std::tuple<unsigned int, dealii::types::global_dof_index, double>> const & wall_times,
  std::string const & operator_type,
  MPI_Comm const &    mpi_comm)
{
  unsigned int N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // clang-format off
    std::cout << std::endl
              << print_horizontal_line()
              << std::endl << std::endl
              << "Operator type: " << operator_type
              << std::endl << std::endl
              << std::setw(5) << std::left << "k"
              << std::setw(15) << std::left << "DoFs"
              << std::setw(15) << std::left << "DoFs/sec"
              << std::setw(15) << std::left << "DoFs/(sec*core)"
              << std::endl << std::flush;

    typedef typename std::vector<std::tuple<unsigned int, dealii::types::global_dof_index, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::setw(5) << std::left << std::get<0>(*it)
                << std::scientific << std::setprecision(4)
                << std::setw(15) << std::left << (double)std::get<1>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)
                << std::setw(15) << std::left << std::get<2>(*it)/(double)N_mpi_processes
                << std::endl << std::flush;
    }

    std::cout << print_horizontal_line() << std::endl << std::endl << std::flush;
    // clang-format on
  }
}

inline void
print_throughput_steady(dealii::ConditionalOStream const &    pcout,
                        dealii::types::global_dof_index const n_dofs,
                        double const                          overall_time_avg,
                        unsigned int const                    N_mpi_processes)
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
print_throughput_10(dealii::ConditionalOStream const &    pcout,
                    dealii::types::global_dof_index const n_dofs,
                    double const                          t_10,
                    unsigned int const                    N_mpi_processes)
{
  double const tau_10 = t_10 * (double)N_mpi_processes / n_dofs;

  // clang-format off
  pcout << std::endl
        << "Throughput of linear solver (numbers based on n_10):" << std::endl
        << "  Number of MPI processes = " << N_mpi_processes << std::endl
        << "  Degrees of freedom      = " << n_dofs << std::endl
        << "  Wall time t_10          = " << std::scientific << std::setprecision(2) << t_10 << " s" << std::endl
        << "  tau_10                  = " << std::scientific << std::setprecision(2) << tau_10 << " s*core/DoF" << std::endl
        << "  Throughput E_10         = " << std::scientific << std::setprecision(2) << 1.0 / tau_10 << " DoF/s/core" << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_throughput_unsteady(dealii::ConditionalOStream const &    pcout,
                          dealii::types::global_dof_index const n_dofs,
                          double const                          overall_time_avg,
                          unsigned int const                    N_time_steps,
                          unsigned int const                    N_mpi_processes)
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
        << "  Throughput              = " << std::scientific << std::setprecision(2) << n_dofs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl
        << std::flush;
  // clang-format on
}


inline void
print_costs(dealii::ConditionalOStream const & pcout,
            double const                       overall_time_avg,
            unsigned int const                 N_mpi_processes)

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
print_solver_info_nonlinear(dealii::ConditionalOStream const & pcout,
                            unsigned int const                 N_iter_nonlinear,
                            unsigned int const                 N_iter_linear,
                            double const                       wall_time)

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
print_solver_info_linear(dealii::ConditionalOStream const & pcout,
                         unsigned int const                 N_iter_linear,
                         double const                       wall_time)

{
  // clang-format off
  pcout << std::endl
        << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
        << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(2) << std::right << wall_time << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_wall_time(dealii::ConditionalOStream const & pcout, double const wall_time)

{
  // clang-format off
  pcout << std::endl
        << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(2) << std::right << wall_time << std::endl
        << std::flush;
  // clang-format on
}

inline void
print_list_of_iterations(dealii::ConditionalOStream const & pcout,
                         std::vector<std::string> const &   names,
                         std::vector<double> const &        iterations_avg)
{
  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  // print
  for(unsigned int i = 0; i < iterations_avg.size(); ++i)
  {
    pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
          << std::setprecision(2) << std::right << std::setw(6) << iterations_avg[i] << std::endl;
  }
}

struct SolverResult
{
  SolverResult() : degree(1), dofs(1), n_10(0), tau_10(0.0)
  {
  }

  SolverResult(unsigned int const                    degree_,
               dealii::types::global_dof_index const dofs_,
               double const                          n_10_,
               double const                          tau_10_)
    : degree(degree_), dofs(dofs_), n_10(n_10_), tau_10(tau_10_)
  {
  }

  static void
  print_header(dealii::ConditionalOStream const & pcout)
  {
    // names
    pcout << std::setw(7) << "degree";
    pcout << std::setw(15) << "dofs";
    pcout << std::setw(8) << "n_10";
    pcout << std::setw(15) << "tau_10";
    pcout << std::setw(15) << "throughput";
    pcout << std::endl;

    // units
    pcout << std::setw(7) << " ";
    pcout << std::setw(15) << " ";
    pcout << std::setw(8) << " ";
    pcout << std::setw(15) << "in s*core/DoF";
    pcout << std::setw(15) << "in DoF/s/core";
    pcout << std::endl;

    pcout << std::endl;
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::setw(7) << std::fixed << degree;
    pcout << std::setw(15) << std::fixed << dofs;
    pcout << std::setw(8) << std::fixed << std::setprecision(1) << n_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << tau_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << 1.0 / tau_10;
    pcout << std::endl;
  }

  unsigned int                    degree;
  dealii::types::global_dof_index dofs;
  double                          n_10;
  double                          tau_10;
};

inline void
print_results(std::vector<SolverResult> const & results, MPI_Comm const & mpi_comm)
{
  // summarize results for all polynomial degrees and problem sizes
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Summary of results:" << std::endl << std::endl;

  SolverResult::print_header(pcout);
  for(std::vector<SolverResult>::const_iterator it = results.begin(); it != results.end(); ++it)
    it->print(pcout);

  pcout << print_horizontal_line() << std::endl << std::endl;
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_PRINT_SOLVER_RESULTS_H_ */
