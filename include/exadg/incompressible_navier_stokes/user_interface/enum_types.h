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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
namespace IncNS
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType refers to the underlying physics of the flow problem and describes
 *  whether the considered flow problem is expected to be a steady or unsteady solution
 *  of the incompressible Navier-Stokes equations. This essentially depends on the
 *  Reynolds number but, for example, also on the boundary conditions (in case of
 *  time dependent boundary conditions, the problem type is always unsteady).
 */
enum class ProblemType
{
  Undefined,
  Steady,
  Unsteady
};

/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., Stokes equations or Navier-Stokes equations
 */
enum class EquationType
{
  Undefined,
  Stokes,
  Euler,
  NavierStokes
};

/*
 *  Formulation of viscous term: divergence formulation or Laplace formulation
 */
enum class FormulationViscousTerm
{
  Undefined,
  DivergenceFormulation,
  LaplaceFormulation
};

/*
 *  Formulation of convective term: divergence formulation or convective formulation
 */
enum class FormulationConvectiveTerm
{
  Undefined,
  DivergenceFormulation,
  ConvectiveFormulation
};

enum class MeshMovementType
{
  Function,
  Poisson,
  Elasticity
};

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  SolverType refers to the numerical solution of the incompressible Navier-Stokes
 *  equations and describes whether a steady or an unsteady solver is used.
 *  While it does not make sense to solve an unsteady problem with a steady solver,
 *  a steady problem can be solved (potentially more efficiently) by using an
 *  unsteady solver.
 */
enum class SolverType
{
  Undefined,
  Steady,
  Unsteady
};

/*
 *  Temporal discretization method
 */
enum class TemporalDiscretization
{
  Undefined,
  BDFDualSplittingScheme,
  BDFPressureCorrection,
  BDFCoupledSolution
};

/*
 *  The convective term can be treated explicitly (Explicit) or implicitly (Implicit).
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,
  Implicit
};

/*
 *  The possibly variable viscosity can be treated explicitly (Explicit) or implicitly (Implicit).
 */
enum class TreatmentOfVariableViscosity
{
  Undefined,
  Explicit,
  Implicit
};

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified,
  CFL,
  MaxEfficiency // only relevant for analytical test cases with optimal rates of
                // convergence in space
};

/*
 *  Pseudo-timestepping for steady-state problems:
 *  Define convergence criterion that is used to terminate simulation
 *
 *  option ResidualSteadyNavierStokes:
 *   - evaluate residual of steady, coupled incompressible Navier-Stokes equations
 *     and terminate simulation if norm of residual fulfills tolerances
 *   - can be used for the coupled solution approach
 *   - can be used for the pressure-correction scheme in case the incremental
 *     formulation is used (for the nonincremental formulation the steady-state
 *     solution cannot fulfill the residual of the steady Navier-Stokes equations
 *     in general due to the splitting error)
 *   - cannot be used for the dual splitting scheme (due to the splitting error
 *     the residual of the steady Navier-Stokes equations is not fulfilled)
 *
 *  option SolutionIncrement:
 *   - calculate solution increment from one time step to the next and terminate
 *     simulation if solution doesn't change any more (defined by tolerances)
 */
enum class ConvergenceCriterionSteadyProblem
{
  Undefined,
  ResidualSteadyNavierStokes,
  SolutionIncrement
};

/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Spatial discretization method.
 *
 *  HDIV implies Raviart-Thomas
 */
enum class SpatialDiscretization
{
  L2,
  HDIV
};

/*
 *  Polynomial degree of pressure shape functions in relation to velocity degree
 */
enum class DegreePressure
{
  MixedOrder,
  EqualOrder
};

/*
 *  Type of imposition of Dirichlet BC's:
 *
 *  direct: u⁺ = g
 *  mirror: u⁺ = -u⁻ + 2g
 *
 *  We normally use the option Mirror as default setup.
 *  A direct imposition might be advantageous with respect to the CFL condition
 *  possibly allowing to use larger time step sizes (approximately 20 percent)
 *  depending on other parameters of the spatial discretization (divergence versus
 *  convective formulation of convective term, upwind factor, and use of divergence
 *  and continuity penalty terms).
 */
enum class TypeDirichletBCs
{
  Direct,
  Mirror
};

/*
 *  Interior penalty formulation of viscous term:
 *  SIPG (symmetric IP) or NIPG (non-symmetric IP)
 *
 *  - use SIPG as default option (sub-optimal rates of convergence observed for NIPG)
 */
enum class InteriorPenaltyFormulation
{
  Undefined,
  SIPG,
  NIPG
};

/*
 *  Penalty term in case of divergence formulation:
 *  not symmetrized: penalty term identical to Laplace formulation, tau * [[u]]
 *  symmetrized: penalty term = tau * ([[u]] + [[u]]^T)
 *
 *  - use Symmetrized as default
 */
enum class PenaltyTermDivergenceFormulation
{
  Undefined,
  Symmetrized,
  NotSymmetrized
};

/*
 * Different options for adjusting the pressure level in case of pure Dirichlet
 * boundary conditions
 *
 * - use ApplyZeroMeanValue as default (this option is always possible)
 */
enum class AdjustPressureLevel
{
  ApplyZeroMeanValue,
  ApplyAnalyticalMeanValue,
  ApplyAnalyticalSolutionInPoint
};

/*
 *  Formulation of velocity divergence term
 */
enum class FormulationVelocityDivergenceTerm
{
  Weak,
  Strong
};

/*
 *  Formulation of pressure gradient term
 */
enum class FormulationPressureGradientTerm
{
  Weak,
  Strong
};

/*
 * Continuity penalty term: apply penalty term to all velocity components or to
 * normal components only.
 *
 * - use Normal as default
 */
enum class ContinuityPenaltyComponents
{
  Undefined,
  All,
  Normal
};

/*
 * Different options for calculation of penalty parameter
 *
 * - use ConvectiveTerm as default
 */
enum class TypePenaltyParameter
{
  Undefined,
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms
};

/**************************************************************************************/
/*                                                                                    */
/*                           NUMERICAL PARAMETERS AND SOLVERS                         */
/*                                                                                    */
/**************************************************************************************/

/*
 * Specify the operator type to be used for multigrid (which can differ from the
 * equation type)
 */
enum class MultigridOperatorType
{
  Undefined,
  ReactionDiffusion,
  ReactionConvectionDiffusion
};

/*
 *  QuadratureRule
 */
enum class QuadratureRuleLinearization
{
  Standard,
  Overintegration32k
};

/**************************************************************************************/
/*                                                                                    */
/*                        HIGH-ORDER DUAL SPLITTING SCHEME                            */
/*                                                                                    */
/**************************************************************************************/


/*
 *  Solver for pressure Poisson equation:
 *
 *  use CG (conjugate gradient) method as default. FGMRES might be necessary
 *  if a Krylov method is used inside the preconditioner (e.g., as multigrid
 *  smoother or as multigrid coarse grid solver)
 */
enum class SolverPressurePoisson
{
  CG,
  FGMRES
};

/*
 *  Preconditioner type for solution of pressure Poisson equation:
 *
 *  use Multigrid as default
 */
enum class PreconditionerPressurePoisson
{
  None,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

/*
 *  Type of projection solver
 *
 *  - use CG as default
 */
enum class SolverProjection
{
  CG,
  FGMRES
};

/*
 *  Preconditioner type for solution of projection step:
 *
 *  use InverseMassMatrix as default. As a rule of thumb, only try other
 *  preconditioners if the number of iterations is significantly larger than 10.
 */
enum class PreconditionerProjection
{
  None,
  InverseMassMatrix,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

/*
 *  Solver type for solution of viscous step:
 *
 *  use CG (conjugate gradient) method as default and GMRES if the problem
 *  is non-symmetric (Divergence formulation of viscous term, but note that often
 *  CG also works in this case).
 *  FGMRES might be necessary if a Krylov method is used inside the preconditioner
 *  (e.g., as multigrid smoother or as multigrid coarse grid solver).
 */
enum class SolverViscous
{
  CG,
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for solution of viscous step:
 *
 *  Use InverseMassMatrix as default. As a rule of thumb, only try other
 *  preconditioners if the number of iterations is significantly larger than 10.
 */
enum class PreconditionerViscous
{
  None,
  InverseMassMatrix,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

/**************************************************************************************/
/*                                                                                    */
/*                             PRESSURE-CORRECTION SCHEME                             */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Solver type for solution of momentum equation
 *
 *  - use CG for symmetric problems and GMRES for non-symmetric problems as default
 *
 *  - FGMRES might be necessary if a Krylov method is used inside the preconditioner
 *    (e.g., as multigrid smoother or as multigrid coarse grid solver).
 */
enum class SolverMomentum
{
  CG,
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for solution of momentum equation:
 *
 *  see coupled solution approach below
 */


/**************************************************************************************/
/*                                                                                    */
/*                            COUPLED NAVIER-STOKES SOLVER                            */
/*                                                                                    */
/**************************************************************************************/

/*
 * Solver for linearized Navier-Stokes problem
 *
 * - use GMRES as default.
 *
 * - FGMRES might be necessary if a Krylov method is used inside the preconditioner
 *   (e.g., as multigrid smoother or as multigrid coarse grid solver).
 */
enum class SolverCoupled
{
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for linearized Navier-Stokes problem
 *
 *  - use BlockTriangular as default (typically best option in terms of time-to-solution, i.e.
 *    BlockDiagonal needs significantly more iterations and BlockTriangularFactorization reduces
 *    number of iterations only slightly but is significantly more expensive)
 */
enum class PreconditionerCoupled
{
  None,
  BlockDiagonal,
  BlockTriangular,
  BlockTriangularFactorization
};

/*
 *  preconditioner for velocity/momentum operator
 *
 *  steady problems:
 *
 *  - use Multigrid as default
 *
 *  unsteady problems:
 *
 *  - use InverseMassMatrix as default. As a rule of thumb, only try other
 *    preconditioners if the number of iterations is significantly larger than 10.
 */
enum class MomentumPreconditioner
{
  None,
  PointJacobi,
  BlockJacobi,
  InverseMassMatrix,
  Multigrid
};

/*
 *  Preconditioner for (2,2) pressure/Schur complement block in case of block preconditioning
 *
 *  default setup:
 *
 *  - InverseMassMatrix for steady Stokes problems
 *  - CahouetChabard for unsteady Stokes problems or unsteady Navier-Stokes problems
 *    with convective term treated explicitly
 *  - PressureConvectionDiffusion for steady/unsteady Navier-Stokes problems with convective term
 *    treated implicitly
 */
enum class SchurComplementPreconditioner
{
  None,
  InverseMassMatrix,
  LaplaceOperator,
  CahouetChabard,
  PressureConvectionDiffusion
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_ */
