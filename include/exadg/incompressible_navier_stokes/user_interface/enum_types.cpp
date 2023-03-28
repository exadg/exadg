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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/user_interface/enum_types.h>

namespace ExaDG
{
namespace IncNS
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(ProblemType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case ProblemType::Undefined:
      string_type = "Undefined";
      break;
    case ProblemType::Steady:
      string_type = "Steady";
      break;
    case ProblemType::Unsteady:
      string_type = "Unsteady";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(EquationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case EquationType::Undefined:
      string_type = "Undefined";
      break;
    case EquationType::Stokes:
      string_type = "Stokes";
      break;
    case EquationType::Euler:
      string_type = "Euler";
      break;
    case EquationType::NavierStokes:
      string_type = "NavierStokes";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(FormulationViscousTerm const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case FormulationViscousTerm::Undefined:
      string_type = "Undefined";
      break;
    case FormulationViscousTerm::DivergenceFormulation:
      string_type = "DivergenceFormulation";
      break;
    case FormulationViscousTerm::LaplaceFormulation:
      string_type = "LaplaceFormulation";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(FormulationConvectiveTerm const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case FormulationConvectiveTerm::Undefined:
      string_type = "Undefined";
      break;
    case FormulationConvectiveTerm::DivergenceFormulation:
      string_type = "DivergenceFormulation";
      break;
    case FormulationConvectiveTerm::ConvectiveFormulation:
      string_type = "ConvectiveFormulation";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(LinearizationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case LinearizationType::Undefined:
      string_type = "Undefined";
      break;
    case LinearizationType::Newton:
      string_type = "Newton";
      break;
    case LinearizationType::Picard:
      string_type = "Picard";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MeshMovementType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MeshMovementType::Function:
      string_type = "Function";
      break;
    case MeshMovementType::Poisson:
      string_type = "Poisson";
      break;
    case MeshMovementType::Elasticity:
      string_type = "Elasticity";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

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

std::string
enum_to_string(TemporalDiscretization const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TemporalDiscretization::Undefined:
      string_type = "Undefined";
      break;
    case TemporalDiscretization::BDFDualSplittingScheme:
      string_type = "BDF dual splitting scheme";
      break;
    case TemporalDiscretization::BDFPressureCorrection:
      string_type = "BDF pressure-correction scheme";
      break;
    case TemporalDiscretization::BDFCoupledSolution:
      string_type = "BDF coupled solution";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TreatmentOfConvectiveTerm const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TreatmentOfConvectiveTerm::Undefined:
      string_type = "Undefined";
      break;
    case TreatmentOfConvectiveTerm::Explicit:
      string_type = "Explicit";
      break;
    case TreatmentOfConvectiveTerm::Implicit:
      string_type = "Implicit";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


std::string
enum_to_string(TimeStepCalculation const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TimeStepCalculation::Undefined:
      string_type = "Undefined";
      break;
    case TimeStepCalculation::UserSpecified:
      string_type = "UserSpecified";
      break;
    case TimeStepCalculation::CFL:
      string_type = "CFL";
      break;
    case TimeStepCalculation::MaxEfficiency:
      string_type = "MaxEfficiency";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(ConvergenceCriterionSteadyProblem const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case ConvergenceCriterionSteadyProblem::Undefined:
      string_type = "Undefined";
      break;
    case ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes:
      string_type = "ResidualSteadyNavierStokes";
      break;
    case ConvergenceCriterionSteadyProblem::SolutionIncrement:
      string_type = "SolutionIncrement";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(DegreePressure const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case DegreePressure::MixedOrder:
      string_type = "Mixed-order";
      break;
    case DegreePressure::EqualOrder:
      string_type = "Equal-order";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(SpatialDiscretization const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SpatialDiscretization::L2:
      string_type = "L2 - Discontinuous Galerkin";
      break;
    case SpatialDiscretization::HDIV:
      string_type = "HDIV - Raviart-Thomas";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TypeDirichletBCs const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TypeDirichletBCs::Direct:
      string_type = "Direct";
      break;
    case TypeDirichletBCs::Mirror:
      string_type = "Mirror";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(InteriorPenaltyFormulation const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case InteriorPenaltyFormulation::Undefined:
      string_type = "Undefined";
      break;
    case InteriorPenaltyFormulation::SIPG:
      string_type = "SIPG";
      break;
    case InteriorPenaltyFormulation::NIPG:
      string_type = "NIPG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PenaltyTermDivergenceFormulation const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PenaltyTermDivergenceFormulation::Undefined:
      string_type = "Undefined";
      break;
    case PenaltyTermDivergenceFormulation::Symmetrized:
      string_type = "Symmetrized";
      break;
    case PenaltyTermDivergenceFormulation::NotSymmetrized:
      string_type = "NotSymmetrized";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(AdjustPressureLevel const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case AdjustPressureLevel::ApplyZeroMeanValue:
      string_type = "ApplyZeroMeanValue";
      break;
    case AdjustPressureLevel::ApplyAnalyticalMeanValue:
      string_type = "ApplyAnalyticalMeanValue";
      break;
    case AdjustPressureLevel::ApplyAnalyticalSolutionInPoint:
      string_type = "ApplyAnalyticalSolutionInPoint";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(FormulationVelocityDivergenceTerm const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case FormulationVelocityDivergenceTerm::Weak:
      string_type = "Weak";
      break;
    case FormulationVelocityDivergenceTerm::Strong:
      string_type = "Strong";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(FormulationPressureGradientTerm const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case FormulationPressureGradientTerm::Weak:
      string_type = "Weak";
      break;
    case FormulationPressureGradientTerm::Strong:
      string_type = "Strong";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(ContinuityPenaltyComponents const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case ContinuityPenaltyComponents::Undefined:
      string_type = "Undefined";
      break;
    case ContinuityPenaltyComponents::All:
      string_type = "All";
      break;
    case ContinuityPenaltyComponents::Normal:
      string_type = "Normal";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TypePenaltyParameter const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TypePenaltyParameter::Undefined:
      string_type = "Undefined";
      break;
    case TypePenaltyParameter::ConvectiveTerm:
      string_type = "ConvectiveTerm";
      break;
    case TypePenaltyParameter::ViscousTerm:
      string_type = "ViscousTerm";
      break;
    case TypePenaltyParameter::ViscousAndConvectiveTerms:
      string_type = "ViscousAndConvectiveTerms";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                           NUMERICAL PARAMETERS AND SOLVERS                         */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(MultigridOperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridOperatorType::Undefined:
      string_type = "Undefined";
      break;
    case MultigridOperatorType::ReactionDiffusion:
      string_type = "ReactionDiffusion";
      break;
    case MultigridOperatorType::ReactionConvectionDiffusion:
      string_type = "ReactionConvectionDiffusion";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(QuadratureRuleLinearization const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case QuadratureRuleLinearization::Standard:
      string_type = "Standard (k+1)";
      break;
    case QuadratureRuleLinearization::Overintegration32k:
      string_type = "Over-integration (3/2k)";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                        HIGH-ORDER DUAL SPLITTING SCHEME                            */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(SolverPressurePoisson const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SolverPressurePoisson::CG:
      string_type = "CG";
      break;
    case SolverPressurePoisson::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerPressurePoisson const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerPressurePoisson::None:
      string_type = "None";
      break;
    case PreconditionerPressurePoisson::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerPressurePoisson::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(SolverProjection const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SolverProjection::CG:
      string_type = "CG";
      break;
    case SolverProjection::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerProjection const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerProjection::None:
      string_type = "None";
      break;
    case PreconditionerProjection::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    case PreconditionerProjection::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerProjection::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case PreconditionerProjection::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(SolverViscous const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SolverViscous::CG:
      string_type = "CG";
      break;
    case SolverViscous::GMRES:
      string_type = "GMRES";
      break;
    case SolverViscous::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerViscous const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerViscous::None:
      string_type = "None";
      break;
    case PreconditionerViscous::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    case PreconditionerViscous::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerViscous::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case PreconditionerViscous::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(SolverMomentum const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SolverMomentum::CG:
      string_type = "CG";
      break;
    case SolverMomentum::GMRES:
      string_type = "GMRES";
      break;
    case SolverMomentum::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(SolverCoupled const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SolverCoupled::GMRES:
      string_type = "GMRES";
      break;
    case SolverCoupled::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerCoupled const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerCoupled::None:
      string_type = "None";
      break;
    case PreconditionerCoupled::BlockDiagonal:
      string_type = "BlockDiagonal";
      break;
    case PreconditionerCoupled::BlockTriangular:
      string_type = "BlockTriangular";
      break;
    case PreconditionerCoupled::BlockTriangularFactorization:
      string_type = "BlockTriangularFactorization";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MomentumPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MomentumPreconditioner::None:
      string_type = "None";
      break;
    case MomentumPreconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case MomentumPreconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case MomentumPreconditioner::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    case MomentumPreconditioner::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(SchurComplementPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SchurComplementPreconditioner::None:
      string_type = "None";
      break;
    case SchurComplementPreconditioner::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    case SchurComplementPreconditioner::LaplaceOperator:
      string_type = "LaplaceOperator";
      break;
    case SchurComplementPreconditioner::CahouetChabard:
      string_type = "CahouetChabard";
      break;
    case SchurComplementPreconditioner::PressureConvectionDiffusion:
      string_type = "PressureConvectionDiffusion";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerMass const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerMass::None:
      string_type = "None";
      break;
    case PreconditionerMass::PointJacobi:
      string_type = "PointJacobi";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                                     TURBULENCE                                     */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(TurbulenceEddyViscosityModel const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TurbulenceEddyViscosityModel::Undefined:
      string_type = "Undefined";
      break;
    case TurbulenceEddyViscosityModel::Smagorinsky:
      string_type = "Smagorinsky";
      break;
    case TurbulenceEddyViscosityModel::Vreman:
      string_type = "Vreman";
      break;
    case TurbulenceEddyViscosityModel::WALE:
      string_type = "WALE";
      break;
    case TurbulenceEddyViscosityModel::Sigma:
      string_type = "Sigma";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace IncNS
} // namespace ExaDG
