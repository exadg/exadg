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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
namespace ConvDiff
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType describes whether a steady or an unsteady problem has to be solved
 */
enum class ProblemType
{
  Undefined,
  Steady,
  Unsteady
};

std::string
enum_to_string(ProblemType const enum_type);

/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., diffusion problem, convective problem or convection-diffusion problem
 */
enum class EquationType
{
  Undefined,
  Convection,
  Diffusion,
  ConvectionDiffusion
};

std::string
enum_to_string(EquationType const enum_type);

/*
 * This parameter describes the type of velocity field for the convective term.
 * Function means that an analytical velocity field is prescribed, while DoFVector
 * means that the velocity field is read and interpolated from a DoFVector, e.g.,
 * the velocity field obtained as the solution of the incompressible Navier-Stokes
 * equations.
 */
enum class TypeVelocityField
{
  Function,
  DoFVector
};

/*
 *  Formulation of convective term: divergence formulation or convective formulation
 */
enum class FormulationConvectiveTerm
{
  DivergenceFormulation,
  ConvectiveFormulation
};

std::string
enum_to_string(FormulationConvectiveTerm const enum_type);

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
 *  Temporal discretization method:
 *  ExplRK: Explicit Runge-Kutta methods (implemented for orders 1-4)
 *  BDF: backward differentiation formulae (implemented for order 1-3)
 */
enum class TemporalDiscretization
{
  Undefined,
  ExplRK,
  BDF
};

std::string
enum_to_string(TemporalDiscretization const enum_type);

/*
 *  For the BDF time integrator, the convective term can be either
 *  treated explicitly or implicitly
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,    // additive decomposition (IMEX)
  ExplicitOIF, // operator-integration-factor splitting (Maday et al. 1990)
  Implicit
};

std::string
enum_to_string(TreatmentOfConvectiveTerm const enum_type);

/*
 *  Temporal discretization method for OIF splitting:
 *
 *    Explicit Runge-Kutta methods
 */
enum class TimeIntegratorRK
{
  Undefined,
  ExplRK1Stage1,
  ExplRK2Stage2,
  ExplRK3Stage3,
  ExplRK4Stage4,
  ExplRK3Stage4Reg2C,
  ExplRK3Stage7Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg2C,
  ExplRK4Stage8Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg3C,
  ExplRK5Stage9Reg2S
};

std::string
enum_to_string(TimeIntegratorRK const enum_type);

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified,
  CFL,
  Diffusion,
  CFLAndDiffusion,
  MaxEfficiency
};

std::string
enum_to_string(TimeStepCalculation const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Numerical flux formulation of convective term
 */
enum class NumericalFluxConvectiveOperator
{
  Undefined,
  CentralFlux,
  LaxFriedrichsFlux
};

std::string
enum_to_string(NumericalFluxConvectiveOperator const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *   Solver for linear system of equations
 */
enum class Solver
{
  Undefined,
  CG,
  GMRES,
  FGMRES // flexible GMRES
};

std::string
enum_to_string(Solver const enum_type);

/*
 *  Preconditioner type for solution of linear system of equations
 */
enum class Preconditioner
{
  Undefined,
  None,
  InverseMassMatrix,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

std::string
enum_to_string(Preconditioner const enum_type);

/*
 * Specify the operator type to be used for multigrid (which can differ from the
 * equation type)
 */
enum class MultigridOperatorType
{
  Undefined,
  ReactionDiffusion,
  ReactionConvection,
  ReactionConvectionDiffusion
};

std::string
enum_to_string(MultigridOperatorType const enum_type);

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace ConvDiff
} // namespace ExaDG


#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_ENUM_TYPES_H_ */
