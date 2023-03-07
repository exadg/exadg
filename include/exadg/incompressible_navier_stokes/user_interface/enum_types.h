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

std::string
enum_to_string(ProblemType const enum_type);

void
string_to_enum(ProblemType & enum_type, std::string const string_type);

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

std::string
enum_to_string(EquationType const enum_type);

void
string_to_enum(EquationType & enum_type, std::string const string_type);

/*
 *  Formulation of viscous term: divergence formulation or Laplace formulation
 */
enum class FormulationViscousTerm
{
  Undefined,
  DivergenceFormulation,
  LaplaceFormulation
};

std::string
enum_to_string(FormulationViscousTerm const enum_type);

void
string_to_enum(FormulationViscousTerm & enum_type, std::string const string_type);

/*
 *  Formulation of convective term: divergence formulation or convective formulation
 */
enum class FormulationConvectiveTerm
{
  Undefined,
  DivergenceFormulation,
  ConvectiveFormulation
};

std::string
enum_to_string(FormulationConvectiveTerm const enum_type);

void
string_to_enum(FormulationConvectiveTerm & enum_type, std::string const string_type);

enum class MeshMovementType
{
  Function,
  Poisson,
  Elasticity
};

std::string
enum_to_string(MeshMovementType const enum_type);

void
string_to_enum(MeshMovementType & enum_type, std::string const string_type);

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

std::string
enum_to_string(TemporalDiscretization const enum_type);

void
string_to_enum(TemporalDiscretization & enum_type, std::string const string_type);

/*
 *  The convective term can be treated explicitly (Explicit) or implicitly (Implicit).
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,
  Implicit
};

std::string
enum_to_string(TreatmentOfConvectiveTerm const enum_type);

void
string_to_enum(TreatmentOfConvectiveTerm & enum_type, std::string const string_type);

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

std::string
enum_to_string(TimeStepCalculation const enum_type);

void
string_to_enum(TimeStepCalculation & enum_type, std::string const string_type);

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

std::string
enum_to_string(ConvergenceCriterionSteadyProblem const enum_type);

void
string_to_enum(ConvergenceCriterionSteadyProblem & enum_type, std::string const string_type);

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

std::string
enum_to_string(SpatialDiscretization const enum_type);

void
string_to_enum(SpatialDiscretization & enum_type, std::string const string_type);

/*
 *  Polynomial degree of pressure shape functions in relation to velocity degree
 */
enum class DegreePressure
{
  MixedOrder,
  EqualOrder
};

std::string
enum_to_string(DegreePressure const enum_type);

void
string_to_enum(DegreePressure & enum_type, std::string const string_type);

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

std::string
enum_to_string(TypeDirichletBCs const enum_type);

void
string_to_enum(TypeDirichletBCs & enum_type, std::string const string_type);

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

std::string
enum_to_string(InteriorPenaltyFormulation const enum_type);

void
string_to_enum(InteriorPenaltyFormulation & enum_type, std::string const string_type);

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

std::string
enum_to_string(PenaltyTermDivergenceFormulation const enum_type);

void
string_to_enum(PenaltyTermDivergenceFormulation & enum_type, std::string const string_type);

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

std::string
enum_to_string(AdjustPressureLevel const enum_type);

void
string_to_enum(AdjustPressureLevel & enum_type, std::string const string_type);

/*
 *  Formulation of velocity divergence term
 */
enum class FormulationVelocityDivergenceTerm
{
  Weak,
  Strong
};

std::string
enum_to_string(FormulationVelocityDivergenceTerm const enum_type);

void
string_to_enum(FormulationVelocityDivergenceTerm & enum_type, std::string const string_type);

/*
 *  Formulation of pressure gradient term
 */
enum class FormulationPressureGradientTerm
{
  Weak,
  Strong
};

std::string
enum_to_string(FormulationPressureGradientTerm const enum_type);

void
string_to_enum(FormulationPressureGradientTerm & enum_type, std::string const string_type);

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

std::string
enum_to_string(ContinuityPenaltyComponents const enum_type);

void
string_to_enum(ContinuityPenaltyComponents & enum_type, std::string const string_type);

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

std::string
enum_to_string(TypePenaltyParameter const enum_type);

void
string_to_enum(TypePenaltyParameter & enum_type, std::string const string_type);

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

std::string
enum_to_string(MultigridOperatorType const enum_type);

void
string_to_enum(MultigridOperatorType & enum_type, std::string const string_type);

/*
 *  QuadratureRule
 */
enum class QuadratureRuleLinearization
{
  Standard,
  Overintegration32k
};

std::string
enum_to_string(QuadratureRuleLinearization const enum_type);

void
string_to_enum(QuadratureRuleLinearization & enum_type, std::string const string_type);

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

std::string
enum_to_string(SolverPressurePoisson const enum_type);

void
string_to_enum(SolverPressurePoisson & enum_type, std::string const string_type);

/*
 *  Preconditioner type for solution of pressure Poisson equation:
 *
 *  use Multigrid as default
 */
enum class PreconditionerPressurePoisson
{
  None,
  PointJacobi,
  Multigrid
};

std::string
enum_to_string(PreconditionerPressurePoisson const enum_type);

void
string_to_enum(PreconditionerPressurePoisson & enum_type, std::string const string_type);

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

std::string
enum_to_string(SolverProjection const enum_type);

void
string_to_enum(SolverProjection & enum_type, std::string const string_type);

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

std::string
enum_to_string(PreconditionerProjection const enum_type);

void
string_to_enum(PreconditionerProjection & enum_type, std::string const string_type);

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

std::string
enum_to_string(SolverViscous const enum_type);

void
string_to_enum(SolverViscous & enum_type, std::string const string_type);

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

std::string
enum_to_string(PreconditionerViscous const enum_type);

void
string_to_enum(PreconditionerViscous & enum_type, std::string const string_type);

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

std::string
enum_to_string(SolverMomentum const enum_type);

void
string_to_enum(SolverMomentum & enum_type, std::string const string_type);

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

std::string
enum_to_string(SolverCoupled const enum_type);

void
string_to_enum(SolverCoupled & enum_type, std::string const string_type);

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

std::string
enum_to_string(PreconditionerCoupled const enum_type);

void
string_to_enum(PreconditionerCoupled & enum_type, std::string const string_type);

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

std::string
enum_to_string(MomentumPreconditioner const enum_type);

void
string_to_enum(MomentumPreconditioner & enum_type, std::string const string_type);

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

std::string
enum_to_string(SchurComplementPreconditioner const enum_type);

void
string_to_enum(SchurComplementPreconditioner & enum_type, std::string const string_type);


/**************************************************************************************/
/*                                                                                    */
/*                            SOLVE MASS SYSTEM (projection)                          */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Preconditioner type for solution of mass system, only used for HDIV case as Matrix-free inverse
 * mass operator is not avaliable.
 *
 *  InverseMassMatrix and BlockJacobi are not availiable.
 */
enum class PreconditionerMass
{
  None,
  PointJacobi
};

std::string
enum_to_string(PreconditionerMass const enum_type);

void
string_to_enum(PreconditionerMass & enum_type, std::string const string_type);

/**************************************************************************************/
/*                                                                                    */
/*                                     TURBULENCE                                     */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Algebraic subgrid-scale turbulence models for LES
 *
 *  Standard constants according to literature:
 *    Smagorinsky: 0.165
 *    Vreman: 0.28
 *    WALE: 0.50
 *    Sigma: 1.35
 */
enum class TurbulenceEddyViscosityModel
{
  Undefined,
  Smagorinsky,
  Vreman,
  WALE,
  Sigma
};

std::string
enum_to_string(TurbulenceEddyViscosityModel const enum_type);

void
string_to_enum(TurbulenceEddyViscosityModel & enum_type, std::string const string_type);


/**************************************************************************************/
/*                                                                                    */
/*                            GENERALIZED NEWTONIAN MODELS                            */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Generalized Newtonian models, see, e.g., Galdi et al., 2008
 *  ("Hemodynamical Flows: Modeling, Analysis and Simulation").
 *  With
 *  y    = sqrt(2*sym_grad_velocity : sym_grad_velocity)
 *  e_oo = generalized_newtonian_kinematic_viscosity_lower_limit
 *  e_0  = generalized_newtonian_kinematic_viscosity_upper_limit
 *  k    = generalized_newtonian_kappa
 *  l    = generalized_newtonian_lambda
 *  a    = generalized_newtonian_a
 *  b    = generalized_newtonian_b
 *  we have the apparent viscosity
 *  viscosity = e_oo + (e_0-e_oo) * [k + (l*y)^a]^[(n-1)/a]
 */
enum class GeneralizedNewtonianModel
{
  Undefined,
  GeneralizedCarreauYasuda,
  Carreau,         // k = 1, a = 2
  Cross,           // k = 1, n = 1 - a
  SimplifiedCross, // k = 1, a = 1, n = 0
  PowerLaw         // e_00 = 0, k = 0
};

std::string
enum_to_string(GeneralizedNewtonianModel const enum_type);

void
string_to_enum(GeneralizedNewtonianModel & enum_type, std::string const string_type);

enum class TreatmentOfNonlinearViscosity
{
  Undefined,
  LinearizedInTimeImplicit,
  Implicit
};

std::string
enum_to_string(TreatmentOfNonlinearViscosity const enum_type);

void
string_to_enum(TreatmentOfNonlinearViscosity & enum_type, std::string const string_type);

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ENUM_TYPES_H_ */
