/*
 * enum_types.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/enum_types.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

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
    case EquationType::Convection:
      string_type = "Convection";
      break;
    case EquationType::Diffusion:
      string_type = "Diffusion";
      break;
    case EquationType::ConvectionDiffusion:
      string_type = "ConvectionDiffusion";
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
    case FormulationConvectiveTerm::DivergenceFormulation:
      string_type = "DivergenceFormulation";
      break;
    case FormulationConvectiveTerm::ConvectiveFormulation:
      string_type = "ConvectiveFormulation";
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
    case TemporalDiscretization::ExplRK:
      string_type = "ExplRK";
      break;
    case TemporalDiscretization::BDF:
      string_type = "BDF";
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
enum_to_string(TimeIntegratorRK const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TimeIntegratorRK::Undefined:
      string_type = "Undefined";
      break;
    case TimeIntegratorRK::ExplRK1Stage1:
      string_type = "ExplRK1Stage1";
      break;
    case TimeIntegratorRK::ExplRK2Stage2:
      string_type = "ExplRK2Stage2";
      break;
    case TimeIntegratorRK::ExplRK3Stage3:
      string_type = "ExplRK3Stage3";
      break;
    case TimeIntegratorRK::ExplRK4Stage4:
      string_type = "ExplRK4Stage4";
      break;
    case TimeIntegratorRK::ExplRK3Stage4Reg2C:
      string_type = "ExplRK3Stage4Reg2C";
      break;
    case TimeIntegratorRK::ExplRK3Stage7Reg2:
      string_type = "ExplRK3Stage7Reg2";
      break;
    case TimeIntegratorRK::ExplRK4Stage5Reg2C:
      string_type = "ExplRK4Stage5Reg2C";
      break;
    case TimeIntegratorRK::ExplRK4Stage8Reg2:
      string_type = "ExplRK4Stage8Reg2";
      break;
    case TimeIntegratorRK::ExplRK4Stage5Reg3C:
      string_type = "ExplRK4Stage5Reg3C";
      break;
    case TimeIntegratorRK::ExplRK5Stage9Reg2S:
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
    case TimeStepCalculation::Diffusion:
      string_type = "Diffusion";
      break;
    case TimeStepCalculation::CFLAndDiffusion:
      string_type = "CFLAndDiffusion";
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

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(NumericalFluxConvectiveOperator const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case NumericalFluxConvectiveOperator::Undefined:
      string_type = "Undefined";
      break;
    case NumericalFluxConvectiveOperator::CentralFlux:
      string_type = "CentralFlux";
      break;
    case NumericalFluxConvectiveOperator::LaxFriedrichsFlux:
      string_type = "LaxFriedrichsFlux";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(Solver const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case Solver::CG:
      string_type = "CG";
      break;
    case Solver::GMRES:
      string_type = "GMRES";
      break;
    case Solver::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(Preconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case Preconditioner::Undefined:
      string_type = "Undefined";
      break;
    case Preconditioner::None:
      string_type = "None";
      break;
    case Preconditioner::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    case Preconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case Preconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case Preconditioner::Multigrid:
      string_type = "Multigrid";
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
    case MultigridOperatorType::ReactionConvection:
      string_type = "ReactionConvection";
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
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace ConvDiff
} // namespace ExaDG
