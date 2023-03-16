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

#ifndef INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_
#define INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_

#include <deal.II/base/conditional_ostream.h>

namespace ExaDG
{
struct SolverResult
{
  SolverResult() : degree(1), dofs(1), n_10(0), tau_10(0.0), print_10(false)
  {
  }

  SolverResult(unsigned int const degree_, dealii::types::global_dof_index const dofs_)
    : degree(degree_), dofs(dofs_), print_10(false)
  {
  }

  SolverResult(unsigned int const                    degree_,
               dealii::types::global_dof_index const dofs_,
               double const                          n_10_,
               double const                          tau_10_)
    : degree(degree_), dofs(dofs_), n_10(n_10_), tau_10(tau_10_), print_10(true)
  {
  }

  void
  print_header(dealii::ConditionalOStream const & pcout) const
  {
    // names
    pcout << std::setw(7) << "degree";
    pcout << std::setw(15) << "dofs";
    if(print_10)
    {
      pcout << std::setw(8) << "n_10";
      pcout << std::setw(15) << "tau_10";
      pcout << std::setw(15) << "throughput";
    }
    pcout << std::endl;

    // units
    pcout << std::setw(7) << " ";
    pcout << std::setw(15) << " ";
    if(print_10)
    {
      pcout << std::setw(8) << " ";
      pcout << std::setw(15) << "in s*core/DoF";
      pcout << std::setw(15) << "in DoF/s/core";
    }
    pcout << std::endl;

    pcout << std::endl;
  }

  void
  print_results(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::setw(7) << std::fixed << degree;
    pcout << std::setw(15) << std::fixed << dofs;
    if(print_10)
    {
      pcout << std::setw(8) << std::fixed << std::setprecision(1) << n_10;
      pcout << std::setw(15) << std::scientific << std::setprecision(2) << tau_10;
      pcout << std::setw(15) << std::scientific << std::setprecision(2) << 1.0 / tau_10;
    }
    pcout << std::endl;
  }

  unsigned int                    degree;
  dealii::types::global_dof_index dofs;
  double                          n_10;
  double                          tau_10;
  bool                            print_10;
};

inline void
print_results(std::vector<SolverResult> const & results, MPI_Comm const & mpi_comm)
{
  // summarize results for all polynomial degrees and problem sizes
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Summary of results:" << std::endl << std::endl;

  results[0].print_header(pcout);
  for(std::vector<SolverResult>::const_iterator it = results.begin(); it != results.end(); ++it)
    it->print_results(pcout);

  pcout << print_horizontal_line() << std::endl << std::endl;
}
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_SOLVER_RESULT_H_ */
