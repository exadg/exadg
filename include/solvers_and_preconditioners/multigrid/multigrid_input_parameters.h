/*
 * MultigridInputParameters.h
 *
 *  Created on: Jun 20, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_

#include "../../functionalities/print_functions.h"
#include "../mg_coarse/mg_coarse_ml.h"

enum class MultigridType
{
  hMG,
  pMG,
  hpMG,
  phMG
};

enum class MultigridSmoother
{
  Chebyshev,
  ChebyshevNonsymmetricOperator,
  GMRES,
  CG,
  Jacobi
};

/*
 *  Multigrid coarse grid solver
 */
enum class MultigridCoarseGridSolver
{
  Chebyshev,
  ChebyshevNonsymmetricOperator,
  PCG_NoPreconditioner,
  PCG_PointJacobi,
  PCG_BlockJacobi,
  GMRES_NoPreconditioner,
  GMRES_PointJacobi,
  GMRES_BlockJacobi,
  AMG_ML
};

struct ChebyshevSmootherData
{
  ChebyshevSmootherData()
    : smoother_poly_degree(5), smoother_smoothing_range(20), eig_cg_n_iterations(20)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    print_parameter(pcout, "Smoother polynomial degree", smoother_poly_degree);
    print_parameter(pcout, "Smoothing range", smoother_smoothing_range);
    print_parameter(pcout, "Iterations eigenvalue calculation", eig_cg_n_iterations);
  }

  // Sets the polynomial degree of the Chebyshev smoother (Chebyshev accelerated Jacobi smoother)
  double smoother_poly_degree;

  // Sets the smoothing range of the Chebyshev smoother
  double smoother_smoothing_range;

  // number of CG iterations for estimation of eigenvalues
  unsigned int eig_cg_n_iterations;
};

enum class PreconditionerGMRESSmoother
{
  None,
  PointJacobi,
  BlockJacobi
};

struct GMRESSmootherData
{
  GMRESSmootherData() : preconditioner(PreconditionerGMRESSmoother::None), number_of_iterations(5)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    std::string str_preconditioner[] = {"None", "PointJacobi", "BlockJacobi"};

    print_parameter(pcout, "Preconditioner", str_preconditioner[(int)preconditioner]);
    print_parameter(pcout, "Number of iterations", number_of_iterations);
  }

  // use Jacobi method as preconditioner
  PreconditionerGMRESSmoother preconditioner;

  // number of GMRES iterations per smoothing step
  unsigned int number_of_iterations;
};

enum class PreconditionerCGSmoother
{
  None,
  PointJacobi,
  BlockJacobi
};

struct CGSmootherData
{
  CGSmootherData() : preconditioner(PreconditionerCGSmoother::None), number_of_iterations(5)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    std::string str_preconditioner[] = {"None", "PointJacobi", "BlockJacobi"};

    print_parameter(pcout, "Preconditioner", str_preconditioner[(int)preconditioner]);
    print_parameter(pcout, "Number of iterations", number_of_iterations);
  }

  // use Jacobi method as preconditioner
  PreconditionerCGSmoother preconditioner;

  // number of GMRES iterations per smoothing step
  unsigned int number_of_iterations;
};

enum class PreconditionerJacobiSmoother
{
  Undefined,
  PointJacobi,
  BlockJacobi
};

struct JacobiSmootherData
{
  JacobiSmootherData()
    : preconditioner(PreconditionerJacobiSmoother::Undefined),
      number_of_smoothing_steps(5),
      damping_factor(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    std::string str_preconditioner[] = {"Undefined", "PointJacobi", "BlockJacobi"};

    print_parameter(pcout, "Preconditioner", str_preconditioner[(int)preconditioner]);
    print_parameter(pcout, "Number of iterations", number_of_smoothing_steps);
    print_parameter(pcout, "Damping factor", damping_factor);
  }

  // use Jacobi method as preconditioner
  PreconditionerJacobiSmoother preconditioner;

  // number of iterations per smoothing step
  unsigned int number_of_smoothing_steps;

  // damping factor
  double damping_factor;
};

struct MultigridData
{
  MultigridData()
    : smoother(MultigridSmoother::Chebyshev),
      coarse_solver(MultigridCoarseGridSolver::Chebyshev),
      type(MultigridType::hMG)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    std::string str_smoother[] = {
      "Chebyshev", "ChebyshevNonsymmetricOperator", "GMRES", "CG", "Jacobi"};

    print_parameter(pcout, "Multigrid smoother", str_smoother[(int)smoother]);

    if(smoother == MultigridSmoother::Chebyshev ||
       smoother == MultigridSmoother::ChebyshevNonsymmetricOperator)
    {
      chebyshev_smoother_data.print(pcout);
    }
    else if(smoother == MultigridSmoother::GMRES)
    {
      gmres_smoother_data.print(pcout);
    }
    else if(smoother == MultigridSmoother::CG)
    {
      cg_smoother_data.print(pcout);
    }
    else if(smoother == MultigridSmoother::Jacobi)
    {
      jacobi_smoother_data.print(pcout);
    }

    std::string str_coarse_solver[] = {"Chebyshev",
                                       "ChebyshevNonsymmetricOperator",
                                       "PCG - no preconditioner",
                                       "PCG - Point-Jacobi preconditioner",
                                       "PCG - Block-Jacobi preconditioner",
                                       "GMRES - No preconditioner",
                                       "GMRES - Point-Jacobi preconditioner",
                                       "GMRES - Block-Jacobi preconditioner",
                                       "AMG - ML"};

    print_parameter(pcout, "Multigrid coarse grid solver", str_coarse_solver[(int)coarse_solver]);

    if(coarse_solver == MultigridCoarseGridSolver::AMG_ML)
      coarse_ml_data.print(pcout);

    std::string str_type[] = {"h-MG", "p-MG", "hp-MG", "ph-MG"};
    print_parameter(pcout, "Multigrid type", str_type[(int)type]);
  }

  // Type of smoother
  MultigridSmoother smoother;

  // Chebyshev smoother (Chebyshev accelerated Jacobi smoother)
  ChebyshevSmootherData chebyshev_smoother_data;

  // GMRES smoother
  GMRESSmootherData gmres_smoother_data;

  // CG smoother
  CGSmootherData cg_smoother_data;

  // Jacobi smoother
  JacobiSmootherData jacobi_smoother_data;

  // Sets the coarse grid solver
  MultigridCoarseGridSolver coarse_solver;

  // Configuration of AMG settings
  MGCoarseMLData coarse_ml_data;

  // Multigrid type: p-GMG vs. h-GMG
  MultigridType type;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRIDINPUTPARAMETERS_H_ */
