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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_

// C/C++
#include <string>
#include <vector>

// deal.II
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/trilinos_precondition.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
enum class MultigridType
{
  Undefined,
  hMG,
  chMG,
  hcMG,
  cMG,
  pMG,
  cpMG,
  pcMG,
  hpMG,
  chpMG,
  hcpMG,
  hpcMG,
  phMG,
  cphMG,
  pchMG,
  phcMG
};

enum class PSequenceType
{
  GoToOne,
  DecreaseByOne,
  Bisect,
  Manual
};

enum class MultigridSmoother
{
  Chebyshev,
  GMRES,
  CG,
  Jacobi
};

enum class AMGType
{
  ML,
  BoomerAMG
};

enum class MultigridCoarseGridSolver
{
  Chebyshev,
  CG,
  GMRES,
  AMG
};

enum class MultigridCoarseGridPreconditioner
{
  None,
  PointJacobi,
  BlockJacobi,
  AMG
};

struct AMGData
{
  AMGData()
  {
    amg_type = AMGType::ML;

#ifdef DEAL_II_WITH_TRILINOS
    ml_data.smoother_sweeps = 1;
    ml_data.n_cycles        = 1;
    ml_data.smoother_type   = "ILU";
#endif

#ifdef DEAL_II_WITH_PETSC
    boomer_data.n_sweeps_coarse = 1;
    boomer_data.max_iter        = 1;
    boomer_data.relaxation_type_down =
      dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
    boomer_data.relaxation_type_up =
      dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
    boomer_data.relaxation_type_coarse =
      dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
#endif
  };

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "    AMG type", amg_type);

    if(amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      print_parameter(pcout, "    Smoother sweeps", ml_data.smoother_sweeps);
      print_parameter(pcout, "    Number of cycles", ml_data.n_cycles);
      print_parameter(pcout, "    Smoother type", ml_data.smoother_type);
#endif
    }
    else if(amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      print_parameter(pcout, "    Smoother sweeps", boomer_data.n_sweeps_coarse);
      print_parameter(pcout, "    Number of cycles", boomer_data.max_iter);
      print_parameter(pcout, "    Smoother type down", boomer_data.relaxation_type_down);
      print_parameter(pcout, "    Smoother type up", boomer_data.relaxation_type_up);
      print_parameter(pcout, "    Smoother type coarse", boomer_data.relaxation_type_coarse);
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  AMGType amg_type;

#ifdef DEAL_II_WITH_TRILINOS
  dealii::TrilinosWrappers::PreconditionAMG::AdditionalData ml_data;
#endif

#ifdef DEAL_II_WITH_PETSC
  dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData boomer_data;
#endif
};

enum class PreconditionerSmoother
{
  None,
  PointJacobi,
  BlockJacobi,
  AdditiveSchwarz
};

struct SmootherData
{
  SmootherData()
    : smoother(MultigridSmoother::Chebyshev),
      preconditioner(PreconditionerSmoother::PointJacobi),
      iterations(5),
      relaxation_factor(0.8),
      smoothing_range(20),
      iterations_eigenvalue_estimation(20)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Smoother", smoother);
    print_parameter(pcout, "Preconditioner smoother", preconditioner);
    print_parameter(pcout, "Iterations smoother", iterations);

    if(smoother == MultigridSmoother::Jacobi)
    {
      print_parameter(pcout, "Relaxation factor", relaxation_factor);
    }

    if(smoother == MultigridSmoother::Chebyshev)
    {
      print_parameter(pcout, "Smoothing range", smoothing_range);
      print_parameter(pcout, "Iterations eigenvalue estimation", iterations_eigenvalue_estimation);
    }
  }

  // Type of smoother
  MultigridSmoother smoother;

  // Preconditioner used for smoother
  PreconditionerSmoother preconditioner;

  // Number of iterations
  unsigned int iterations;

  // damping/relaxation factor for Jacobi smoother
  double relaxation_factor;

  // Chebyshev smmother: sets the smoothing range (range of eigenvalues to be smoothed)
  double smoothing_range;

  // Chebyshev smmother: number of CG iterations for estimation of eigenvalues
  unsigned int iterations_eigenvalue_estimation;
};

struct CoarseGridData
{
  CoarseGridData()
    : solver(MultigridCoarseGridSolver::Chebyshev),
      preconditioner(MultigridCoarseGridPreconditioner::PointJacobi),
      solver_data(SolverData(1e4, 1.e-12, 1.e-3)),
      amg_data(AMGData())
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Coarse grid solver", solver);
    print_parameter(pcout, "Coarse grid preconditioner", preconditioner);

    solver_data.print(pcout);

    if(solver == MultigridCoarseGridSolver::AMG or
       preconditioner == MultigridCoarseGridPreconditioner::AMG)
    {
      amg_data.print(pcout);
    }
  }

  // Coarse grid solver
  MultigridCoarseGridSolver solver;

  // Coarse grid preconditioner
  MultigridCoarseGridPreconditioner preconditioner;

  // Solver data for coarse grid solver
  SolverData solver_data;

  // Configuration of AMG settings
  AMGData amg_data;
};


struct MultigridData
{
  MultigridData()
    : type(MultigridType::hMG),
      p_sequence(PSequenceType::Bisect),
      smoother_data(SmootherData()),
      coarse_problem(CoarseGridData())
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Multigrid type", type);

    if(involves_p_transfer())
    {
      print_parameter(pcout, "p-sequence", p_sequence);
    }

    smoother_data.print(pcout);

    coarse_problem.print(pcout);
  }

  bool
  involves_h_transfer() const;

  bool
  involves_c_transfer() const;

  bool
  involves_p_transfer() const;

  // Multigrid type: p-MG vs. h-MG
  MultigridType type;

  // Sequence of polynomial degrees during p-multigrid
  PSequenceType p_sequence;

  // Smoother data
  SmootherData smoother_data;

  // Coarse grid problem
  CoarseGridData coarse_problem;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_ */
