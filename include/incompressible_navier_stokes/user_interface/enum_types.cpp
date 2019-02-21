/*
 * enum_types.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "enum_types.h"

using namespace dealii;

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
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case EquationType::NavierStokes:
      string_type = "NavierStokes";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case FormulationConvectiveTerm::EnergyPreservingFormulation:
      string_type = "EnergyPreservingFormulation";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case TreatmentOfConvectiveTerm::ExplicitOIF:
      string_type = "ExplicitOIF";
      break;
    case TreatmentOfConvectiveTerm::Implicit:
      string_type = "Implicit";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TimeIntegratorOIF const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TimeIntegratorOIF::Undefined:
      string_type = "Undefined";
      break;
    case TimeIntegratorOIF::ExplRK1Stage1:
      string_type = "ExplRK1Stage1";
      break;
    case TimeIntegratorOIF::ExplRK2Stage2:
      string_type = "ExplRK2Stage2";
      break;
    case TimeIntegratorOIF::ExplRK3Stage3:
      string_type = "ExplRK3Stage3";
      break;
    case TimeIntegratorOIF::ExplRK4Stage4:
      string_type = "ExplRK4Stage4";
      break;
    case TimeIntegratorOIF::ExplRK3Stage4Reg2C:
      string_type = "ExplRK3Stage4Reg2C";
      break;
    case TimeIntegratorOIF::ExplRK3Stage7Reg2:
      string_type = "ExplRK3Stage7Reg2";
      break;
    case TimeIntegratorOIF::ExplRK4Stage5Reg2C:
      string_type = "ExplRK4Stage5Reg2C";
      break;
    case TimeIntegratorOIF::ExplRK4Stage8Reg2:
      string_type = "ExplRK4Stage8Reg2";
      break;
    case TimeIntegratorOIF::ExplRK4Stage5Reg3C:
      string_type = "ExplRK4Stage5Reg3C";
      break;
    case TimeIntegratorOIF::ExplRK5Stage9Reg2S:
      string_type = "ExplRK5Stage9Reg2S";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TriangulationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TriangulationType::Undefined:
      string_type = "Undefined";
      break;
    case TriangulationType::Distributed:
      string_type = "Distributed";
      break;
    case TriangulationType::FullyDistributed:
      string_type = "FullyDistributed";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
enum_to_string(PreconditionerBlockDiagonal const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerBlockDiagonal::Undefined:
      string_type = "Undefined";
      break;
    case PreconditionerBlockDiagonal::None:
      string_type = "None";
      break;
    case PreconditionerBlockDiagonal::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case SolverProjection::LU:
      string_type = "LU";
      break;
    case SolverProjection::CG:
      string_type = "CG";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case PreconditionerProjection::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerProjection::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case PreconditionerProjection::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
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
    case SchurComplementPreconditioner::Elman:
      string_type = "Elman";
      break;
    case SchurComplementPreconditioner::PressureConvectionDiffusion:
      string_type = "PressureConvectionDiffusion";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(DiscretizationOfLaplacian const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case DiscretizationOfLaplacian::Classical:
      string_type = "Classical";
      break;
    case DiscretizationOfLaplacian::Compatible:
      string_type = "Compatible";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
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
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace IncNS
