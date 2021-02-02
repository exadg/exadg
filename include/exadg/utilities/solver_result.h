/*
 * solver_result.h
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_
#define INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_

#include <deal.II/base/conditional_ostream.h>

namespace ExaDG
{
using namespace dealii;

struct SolverResult
{
  SolverResult() : degree(1), dofs(1), n_10(0), tau_10(0.0)
  {
  }

  SolverResult(unsigned int const            degree_,
               types::global_dof_index const dofs_,
               double const                  n_10_,
               double const                  tau_10_)
    : degree(degree_), dofs(dofs_), n_10(n_10_), tau_10(tau_10_)
  {
  }

  void
  print_header(ConditionalOStream const & pcout) const
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
  print_results(ConditionalOStream const & pcout) const
  {
    pcout << std::setw(7) << std::fixed << degree;
    pcout << std::setw(15) << std::fixed << dofs;
    pcout << std::setw(8) << std::fixed << std::setprecision(1) << n_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << tau_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << 1.0 / tau_10;
    pcout << std::endl;
  }

  unsigned int            degree;
  types::global_dof_index dofs;
  double                  n_10;
  double                  tau_10;
};

inline void
print_results(std::vector<SolverResult> const & results, MPI_Comm const & mpi_comm)
{
  // summarize results for all polynomial degrees and problem sizes
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Summary of results:" << std::endl << std::endl;

  results[0].print_header(pcout);
  for(std::vector<SolverResult>::const_iterator it = results.begin(); it != results.end(); ++it)
    it->print_results(pcout);

  pcout << print_horizontal_line() << std::endl << std::endl;
}
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_ */
