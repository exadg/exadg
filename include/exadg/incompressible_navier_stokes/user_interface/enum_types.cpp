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

void
string_to_enum(ProblemType & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = ProblemType::Undefined;
  else if(string_type == "Steady")
    enum_type = ProblemType::Steady;
  else if(string_type == "Unsteady")
    enum_type = ProblemType::Unsteady;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(EquationType & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = EquationType::Undefined;
  else if(string_type == "Stokes")
    enum_type = EquationType::Stokes;
  else if(string_type == "NavierStokes")
    enum_type = EquationType::NavierStokes;
  else if(string_type == "Euler")
    enum_type = EquationType::Euler;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(FormulationViscousTerm & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = FormulationViscousTerm::Undefined;
  else if(string_type == "LaplaceFormulation")
    enum_type = FormulationViscousTerm::LaplaceFormulation;
  else if(string_type == "DivergenceFormulation")
    enum_type = FormulationViscousTerm::DivergenceFormulation;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(FormulationConvectiveTerm & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = FormulationConvectiveTerm::Undefined;
  else if(string_type == "DivergenceFormulation")
    enum_type = FormulationConvectiveTerm::DivergenceFormulation;
  else if(string_type == "ConvectiveFormulation")
    enum_type = FormulationConvectiveTerm::ConvectiveFormulation;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(MeshMovementType & enum_type, std::string const string_type)
{
  if(string_type == "Function")
    enum_type = MeshMovementType::Function;
  else if(string_type == "Poisson")
    enum_type = MeshMovementType::Poisson;
  else if(string_type == "Elasticity")
    enum_type = MeshMovementType::Elasticity;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TemporalDiscretization & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TemporalDiscretization::Undefined;
  else if(string_type == "BDFDualSplittingScheme")
    enum_type = TemporalDiscretization::BDFDualSplittingScheme;
  else if(string_type == "BDFPressureCorrection")
    enum_type = TemporalDiscretization::BDFPressureCorrection;
  else if(string_type == "BDFCoupledSolution")
    enum_type = TemporalDiscretization::BDFCoupledSolution;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TreatmentOfConvectiveTerm & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TreatmentOfConvectiveTerm::Undefined;
  else if(string_type == "Explicit")
    enum_type = TreatmentOfConvectiveTerm::Explicit;
  else if(string_type == "Implicit")
    enum_type = TreatmentOfConvectiveTerm::Implicit;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TimeStepCalculation & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TimeStepCalculation::Undefined;
  else if(string_type == "UserSpecified")
    enum_type = TimeStepCalculation::UserSpecified;
  else if(string_type == "CFL")
    enum_type = TimeStepCalculation::CFL;
  else if(string_type == "MaxEfficiency")
    enum_type = TimeStepCalculation::MaxEfficiency;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(ConvergenceCriterionSteadyProblem & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = ConvergenceCriterionSteadyProblem::Undefined;
  else if(string_type == "ResidualSteadyNavierStokes")
    enum_type = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  else if(string_type == "SolutionIncrement")
    enum_type = ConvergenceCriterionSteadyProblem::SolutionIncrement;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SpatialDiscretization & enum_type, std::string const string_type)
{
  if(string_type == "L2")
    enum_type = SpatialDiscretization::L2;
  else if(string_type == "HDIV")
    enum_type = SpatialDiscretization::HDIV;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(DegreePressure & enum_type, std::string const string_type)
{
  if(string_type == "MixedOrder")
    enum_type = DegreePressure::MixedOrder;
  else if(string_type == "EqualOrder")
    enum_type = DegreePressure::EqualOrder;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TypeDirichletBCs & enum_type, std::string const string_type)
{
  if(string_type == "Direct")
    enum_type = TypeDirichletBCs::Direct;
  else if(string_type == "Mirror")
    enum_type = TypeDirichletBCs::Mirror;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(InteriorPenaltyFormulation & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = InteriorPenaltyFormulation::Undefined;
  else if(string_type == "SIPG")
    enum_type = InteriorPenaltyFormulation::SIPG;
  else if(string_type == "NIPG")
    enum_type = InteriorPenaltyFormulation::NIPG;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PenaltyTermDivergenceFormulation & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = PenaltyTermDivergenceFormulation::Undefined;
  else if(string_type == "Symmetrized")
    enum_type = PenaltyTermDivergenceFormulation::Symmetrized;
  else if(string_type == "NotSymmetrized")
    enum_type = PenaltyTermDivergenceFormulation::NotSymmetrized;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(AdjustPressureLevel & enum_type, std::string const string_type)
{
  if(string_type == "ApplyZeroMeanValue")
    enum_type = AdjustPressureLevel::ApplyZeroMeanValue;
  else if(string_type == "ApplyAnalyticalMeanValue")
    enum_type = AdjustPressureLevel::ApplyAnalyticalMeanValue;
  else if(string_type == "ApplyAnalyticalSolutionInPoint")
    enum_type = AdjustPressureLevel::ApplyAnalyticalSolutionInPoint;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(FormulationVelocityDivergenceTerm & enum_type, std::string const string_type)
{
  if(string_type == "Weak")
    enum_type = FormulationVelocityDivergenceTerm::Weak;
  else if(string_type == "Strong")
    enum_type = FormulationVelocityDivergenceTerm::Strong;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(FormulationPressureGradientTerm & enum_type, std::string const string_type)
{
  if(string_type == "Weak")
    enum_type = FormulationPressureGradientTerm::Weak;
  else if(string_type == "Strong")
    enum_type = FormulationPressureGradientTerm::Strong;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(ContinuityPenaltyComponents & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = ContinuityPenaltyComponents::Undefined;
  else if(string_type == "All")
    enum_type = ContinuityPenaltyComponents::All;
  else if(string_type == "Normal")
    enum_type = ContinuityPenaltyComponents::Normal;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TypePenaltyParameter & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TypePenaltyParameter::Undefined;
  else if(string_type == "ConvectiveTerm")
    enum_type = TypePenaltyParameter::ConvectiveTerm;
  else if(string_type == "ViscousTerm")
    enum_type = TypePenaltyParameter::ViscousTerm;
  else if(string_type == "ViscousAndConvectiveTerms")
    enum_type = TypePenaltyParameter::ViscousAndConvectiveTerms;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(MultigridOperatorType & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = MultigridOperatorType::Undefined;
  else if(string_type == "ReactionDiffusion")
    enum_type = MultigridOperatorType::ReactionDiffusion;
  else if(string_type == "ReactionConvectionDiffusion")
    enum_type = MultigridOperatorType::ReactionConvectionDiffusion;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(QuadratureRuleLinearization & enum_type, std::string const string_type)
{
  if(string_type == "Standard")
    enum_type = QuadratureRuleLinearization::Standard;
  else if(string_type == "Overintegration32k")
    enum_type = QuadratureRuleLinearization::Overintegration32k;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SolverPressurePoisson & enum_type, std::string const string_type)
{
  if(string_type == "CG")
    enum_type = SolverPressurePoisson::CG;
  else if(string_type == "FGMRES")
    enum_type = SolverPressurePoisson::FGMRES;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PreconditionerPressurePoisson & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = PreconditionerPressurePoisson::None;
  else if(string_type == "PointJacobi")
    enum_type = PreconditionerPressurePoisson::PointJacobi;
  else if(string_type == "Multigrid")
    enum_type = PreconditionerPressurePoisson::Multigrid;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SolverProjection & enum_type, std::string const string_type)
{
  if(string_type == "CG")
    enum_type = SolverProjection::CG;
  else if(string_type == "FGMRES")
    enum_type = SolverProjection::FGMRES;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PreconditionerProjection & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = PreconditionerProjection::None;
  else if(string_type == "InverseMassMatrix")
    enum_type = PreconditionerProjection::InverseMassMatrix;
  else if(string_type == "PointJacobi")
    enum_type = PreconditionerProjection::PointJacobi;
  else if(string_type == "BlockJacobi")
    enum_type = PreconditionerProjection::BlockJacobi;
  else if(string_type == "Multigrid")
    enum_type = PreconditionerProjection::Multigrid;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SolverViscous & enum_type, std::string const string_type)
{
  if(string_type == "CG")
    enum_type = SolverViscous::CG;
  else if(string_type == "GMRES")
    enum_type = SolverViscous::GMRES;
  else if(string_type == "FGMRES")
    enum_type = SolverViscous::FGMRES;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PreconditionerViscous & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = PreconditionerViscous::None;
  else if(string_type == "InverseMassMatrix")
    enum_type = PreconditionerViscous::InverseMassMatrix;
  else if(string_type == "PointJacobi")
    enum_type = PreconditionerViscous::PointJacobi;
  else if(string_type == "BlockJacobi")
    enum_type = PreconditionerViscous::BlockJacobi;
  else if(string_type == "Multigrid")
    enum_type = PreconditionerViscous::Multigrid;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SolverMomentum & enum_type, std::string const string_type)
{
  if(string_type == "CG")
    enum_type = SolverMomentum::CG;
  else if(string_type == "GMRES")
    enum_type = SolverMomentum::GMRES;
  else if(string_type == "FGMRES")
    enum_type = SolverMomentum::FGMRES;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SolverCoupled & enum_type, std::string const string_type)
{
  if(string_type == "GMRES")
    enum_type = SolverCoupled::GMRES;
  else if(string_type == "FGMRES")
    enum_type = SolverCoupled::FGMRES;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PreconditionerCoupled & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = PreconditionerCoupled::None;
  else if(string_type == "BlockDiagonal")
    enum_type = PreconditionerCoupled::BlockDiagonal;
  else if(string_type == "BlockTriangular")
    enum_type = PreconditionerCoupled::BlockTriangular;
  else if(string_type == "BlockTriangularFactorization")
    enum_type = PreconditionerCoupled::BlockTriangularFactorization;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(MomentumPreconditioner & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = MomentumPreconditioner::None;
  else if(string_type == "PointJacobi")
    enum_type = MomentumPreconditioner::PointJacobi;
  else if(string_type == "BlockJacobi")
    enum_type = MomentumPreconditioner::BlockJacobi;
  else if(string_type == "InverseMassMatrix")
    enum_type = MomentumPreconditioner::InverseMassMatrix;
  else if(string_type == "Multigrid")
    enum_type = MomentumPreconditioner::Multigrid;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(SchurComplementPreconditioner & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = SchurComplementPreconditioner::None;
  else if(string_type == "InverseMassMatrix")
    enum_type = SchurComplementPreconditioner::InverseMassMatrix;
  else if(string_type == "LaplaceOperator")
    enum_type = SchurComplementPreconditioner::LaplaceOperator;
  else if(string_type == "CahouetChabard")
    enum_type = SchurComplementPreconditioner::CahouetChabard;
  else if(string_type == "PressureConvectionDiffusion")
    enum_type = SchurComplementPreconditioner::PressureConvectionDiffusion;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(PreconditionerMass & enum_type, std::string const string_type)
{
  if(string_type == "None")
    enum_type = PreconditionerMass::None;
  else if(string_type == "PointJacobi")
    enum_type = PreconditionerMass::PointJacobi;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
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

void
string_to_enum(TurbulenceEddyViscosityModel & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TurbulenceEddyViscosityModel::Undefined;
  else if(string_type == "Smagorinsky")
    enum_type = TurbulenceEddyViscosityModel::Smagorinsky;
  else if(string_type == "Vreman")
    enum_type = TurbulenceEddyViscosityModel::Vreman;
  else if(string_type == "WALE")
    enum_type = TurbulenceEddyViscosityModel::WALE;
  else if(string_type == "Sigma")
    enum_type = TurbulenceEddyViscosityModel::Sigma;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
}

/**************************************************************************************/
/*                                                                                    */
/*                            GENERALIZED NEWTONIAN MODELS                            */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(GeneralizedNewtonianModel const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case GeneralizedNewtonianModel::Undefined:
      string_type = "Undefined";
      break;
    case GeneralizedNewtonianModel::GeneralizedCarreauYasuda:
      string_type = "GeneralizedCarreauYasuda";
      break;
    case GeneralizedNewtonianModel::Carreau:
      string_type = "Carreau";
      break;
    case GeneralizedNewtonianModel::Cross:
      string_type = "Cross";
      break;
    case GeneralizedNewtonianModel::SimplifiedCross:
      string_type = "SimplifiedCross";
      break;
    case GeneralizedNewtonianModel::PowerLaw:
      string_type = "PowerLaw";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

void
string_to_enum(GeneralizedNewtonianModel & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = GeneralizedNewtonianModel::Undefined;
  else if(string_type == "GeneralizedCarreauYasuda")
    enum_type = GeneralizedNewtonianModel::GeneralizedCarreauYasuda;
  else if(string_type == "Carreau")
    enum_type = GeneralizedNewtonianModel::Carreau;
  else if(string_type == "Cross")
    enum_type = GeneralizedNewtonianModel::Cross;
  else if(string_type == "SimplifiedCross")
    enum_type = GeneralizedNewtonianModel::SimplifiedCross;
  else if(string_type == "PowerLaw")
    enum_type = GeneralizedNewtonianModel::PowerLaw;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
}

std::string
enum_to_string(TreatmentOfNonlinearViscosity const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TreatmentOfNonlinearViscosity::Undefined:
      string_type = "Undefined";
      break;
    case TreatmentOfNonlinearViscosity::LinearizedInTimeImplicit:
      string_type = "LinearizedInTimeImplicit";
      break;
    case TreatmentOfNonlinearViscosity::Implicit:
      string_type = "Implicit";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

void
string_to_enum(TreatmentOfNonlinearViscosity & enum_type, std::string const string_type)
{
  if(string_type == "Undefined")
    enum_type = TreatmentOfNonlinearViscosity::Undefined;
  else if(string_type == "LinearizedInTimeImplicit")
    enum_type = TreatmentOfNonlinearViscosity::LinearizedInTimeImplicit;
  else if(string_type == "Implicit")
    enum_type = TreatmentOfNonlinearViscosity::Implicit;
  else
    AssertThrow(false, dealii::ExcMessage("Unknown enum type. Not implemented."));
}

} // namespace IncNS
} // namespace ExaDG
