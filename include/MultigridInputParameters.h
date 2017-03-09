/*
 * MultigridInputParameters.h
 *
 *  Created on: Jun 20, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_MULTIGRIDINPUTPARAMETERS_H_

#include "PrintFunctions.h"

 enum class MultigridSmoother
 {
   Chebyshev,
   ChebyshevNonsymmetricOperator,
   GMRES,
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
  PCG_Jacobi,
  GMRES_NoPreconditioner,
  GMRES_Jacobi
};

struct ChebyshevSmootherData
{
  ChebyshevSmootherData()
    :
    smoother_poly_degree(5),
    smoother_smoothing_range(20),
    eig_cg_n_iterations(20)
    {}

  void print(ConditionalOStream &pcout)
  {
    print_parameter(pcout,"Smoother polynomial degree",smoother_poly_degree);
    print_parameter(pcout,"Smoothing range",smoother_smoothing_range);
    print_parameter(pcout,"Iterations eigenvalue calculation",eig_cg_n_iterations);
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
  GMRESSmootherData()
    :
    preconditioner(PreconditionerGMRESSmoother::None),
    number_of_iterations(5)
  {}

  void print(ConditionalOStream &pcout)
  {
    std::string str_preconditioner[] = { "None",
                                         "PointJacobi",
                                         "BlockJacobi"};

    print_parameter(pcout,"Preconditioner",str_preconditioner[(int)preconditioner]);
    print_parameter(pcout,"Number of iterations",number_of_iterations);
  }

  // use Jacobi method as preconditioner
  PreconditionerGMRESSmoother preconditioner;

  // number of GMRES iterations per smoothing step
  unsigned int number_of_iterations;

};

enum class PreconditionerJacobiSmoother
{
  None,
  PointJacobi,
  BlockJacobi
};

struct JacobiSmootherData
{
  JacobiSmootherData()
    :
    preconditioner(PreconditionerJacobiSmoother::None),
    number_of_smoothing_steps(5),
    damping_factor(1.0)
  {}

  void print(ConditionalOStream &pcout)
  {
    std::string str_preconditioner[] = { "None",
                                         "PointJacobi",
                                         "BlockJacobi"};

    print_parameter(pcout,"Preconditioner",str_preconditioner[(int)preconditioner]);
    print_parameter(pcout,"Number of iterations",number_of_smoothing_steps);
    print_parameter(pcout,"Damping factor",damping_factor);
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
    :
    smoother(MultigridSmoother::Chebyshev),
    coarse_solver(MultigridCoarseGridSolver::Chebyshev)
  {}

  void print(ConditionalOStream &pcout)
  {
    std::string str_smoother[] = { "Chebyshev",
                                   "ChebyshevNonsymmetricOperator",
                                   "GMRES",
                                   "Jacobi"};

    print_parameter(pcout,"Multigrid smoother",str_smoother[(int)smoother]);

    if(smoother == MultigridSmoother::Chebyshev || smoother == MultigridSmoother::ChebyshevNonsymmetricOperator)
    {
      chebyshev_smoother_data.print(pcout);
    }
    else if(smoother == MultigridSmoother::GMRES)
    {
      gmres_smoother_data.print(pcout);
    }
    else if(smoother == MultigridSmoother::Jacobi)
    {
      jacobi_smoother_data.print(pcout);
    }

    std::string str_coarse_solver[] = { "Chebyshev",
                                        "ChebyshevNonsymmetricOperator",
                                        "PCG - no preconditioner",
                                        "PCG - Jacobi preconditioner",
                                        "GMRES - No preconditioner",
                                        "GMRES - Jacobi preconditioner"};

    print_parameter(pcout,"Multigrid coarse grid solver",str_coarse_solver[(int)coarse_solver]);

  }

  // Type of smoother
  MultigridSmoother smoother;

  // Chebyshev smoother (Chebyshev accelerated Jacobi smoother)
  ChebyshevSmootherData chebyshev_smoother_data;

  // GMRES smoother
  GMRESSmootherData gmres_smoother_data;

  // Jacobi smoother
  JacobiSmootherData jacobi_smoother_data;

  // Sets the coarse grid solver
  MultigridCoarseGridSolver coarse_solver;
};


#endif /* INCLUDE_MULTIGRIDINPUTPARAMETERS_H_ */
