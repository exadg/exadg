/*
 * multigrid_input_parameters.h
 *
 *  Created on: Jun 20, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_

#include <string>
#include <vector>

// deal.II
#include <deal.II/lac/trilinos_precondition.h>

#include "../solvers/solver_data.h"

#include "../../functionalities/print_functions.h"

enum class MultigridType
{
  Undefined,
  hMG,
  chMG,
  hcMG,
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

std::string
enum_to_string(MultigridType const enum_type);


enum class PSequenceType
{
  GoToOne,
  DecreaseByOne,
  Bisect,
  Manual
};

std::string
enum_to_string(PSequenceType const enum_type);

enum class MultigridSmoother
{
  Chebyshev,
  GMRES,
  CG,
  Jacobi
};

std::string
enum_to_string(MultigridSmoother const enum_type);


enum class MultigridCoarseGridSolver
{
  Chebyshev,
  CG,
  GMRES,
  AMG
};

std::string
enum_to_string(MultigridCoarseGridSolver const enum_type);


enum class MultigridCoarseGridPreconditioner
{
  None,
  PointJacobi,
  BlockJacobi,
  AMG
};

std::string
enum_to_string(MultigridCoarseGridPreconditioner const enum_type);

#ifndef DEAL_II_WITH_TRILINOS
namespace dealii
{
namespace TrilinosWrappers
{
namespace PreconditionAMG
{
// copy of interface from deal.II (lac/trilinos_precondition.h)
struct AdditionalData
{
  AdditionalData(
    const bool                           elliptic              = true,
    const bool                           higher_order_elements = false,
    const unsigned int                   n_cycles              = 1,
    const bool                           w_cycle               = false,
    const double                         aggregation_threshold = 1e-4,
    const std::vector<std::vector<bool>> constant_modes        = std::vector<std::vector<bool>>(0),
    const unsigned int                   smoother_sweeps       = 2,
    const unsigned int                   smoother_overlap      = 0,
    const bool                           output_details        = false,
    const char *                         smoother_type         = "Chebyshev",
    const char *                         coarse_type           = "Amesos-KLU")
    : elliptic(elliptic),
      higher_order_elements(higher_order_elements),
      n_cycles(n_cycles),
      w_cycle(w_cycle),
      aggregation_threshold(aggregation_threshold),
      constant_modes(constant_modes),
      smoother_sweeps(smoother_sweeps),
      smoother_overlap(smoother_overlap),
      output_details(output_details),
      smoother_type(smoother_type),
      coarse_type(coarse_type)
  {
  }

  bool                           elliptic;
  bool                           higher_order_elements;
  unsigned int                   n_cycles;
  bool                           w_cycle;
  double                         aggregation_threshold;
  std::vector<std::vector<bool>> constant_modes;
  unsigned int                   smoother_sweeps;
  unsigned int                   smoother_overlap;
  bool                           output_details;
  const char *                   smoother_type;
  const char *                   coarse_type;
};

} // namespace PreconditionAMG
} // namespace TrilinosWrappers
} // namespace dealii
#endif

struct AMGData
{
  AMGData()
  {
    data.smoother_sweeps = 1;
    data.n_cycles        = 1;
    data.smoother_type   = "ILU";
  };

  void
  print(ConditionalOStream & pcout)
  {
    print_parameter(pcout, "    Smoother sweeps", data.smoother_sweeps);
    print_parameter(pcout, "    Number of cycles", data.n_cycles);
    print_parameter(pcout, "    Smoother type", data.smoother_type);
  }

  TrilinosWrappers::PreconditionAMG::AdditionalData data;
};

enum class PreconditionerSmoother
{
  None,
  PointJacobi,
  BlockJacobi
};

std::string
enum_to_string(PreconditionerSmoother const enum_type);

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
  print(ConditionalOStream & pcout)
  {
    print_parameter(pcout, "Smoother", enum_to_string(smoother));
    print_parameter(pcout, "Preconditioner smoother", enum_to_string(preconditioner));
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

  // number of CG iterations for estimation of eigenvalues
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
  print(ConditionalOStream & pcout)
  {
    print_parameter(pcout, "Coarse grid solver", enum_to_string(solver));
    print_parameter(pcout, "Coarse grid preconditioner", enum_to_string(preconditioner));

    solver_data.print(pcout);

    if(solver == MultigridCoarseGridSolver::AMG ||
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
  print(ConditionalOStream & pcout)
  {
    print_parameter(pcout, "Multigrid type", enum_to_string(type));

    if(type != MultigridType::hMG || type != MultigridType::hcMG || type != MultigridType::chMG)
    {
      print_parameter(pcout, "p-sequence", enum_to_string(p_sequence));
    }

    smoother_data.print(pcout);

    coarse_problem.print(pcout);
  }

  // Multigrid type: p-MG vs. h-MG
  MultigridType type;

  // Sequence of polynomial degrees during p-multigrid
  PSequenceType p_sequence;

  // Smoother data
  SmootherData smoother_data;

  // Coarse grid problem
  CoarseGridData coarse_problem;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_ */
