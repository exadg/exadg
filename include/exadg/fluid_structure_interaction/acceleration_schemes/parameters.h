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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

// ExaDG
#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
namespace FSI
{
enum class AccelerationMethod
{
  Undefined,
  Aitken,
  IQN_ILS,
  IQN_IMVLS
};

// The initial guess used in the FSI coupling loop is computed using an extrapolation of suitable
// order (defined within the single-field time integrators) or the last iterate from the previous
// time step.
enum class InitialGuessCouplingScheme
{
  SolutionExtrapolatedToEndOfTimeStep,
  ConvergedSolutionOfPreviousTimeStep
};

struct Parameters
{
  Parameters()
    : acceleration_method(AccelerationMethod::Undefined),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      initial_guess_coupling_scheme(
        InitialGuessCouplingScheme::SolutionExtrapolatedToEndOfTimeStep),
      reused_time_steps(0),
      partitioned_iter_max(100),
      geometric_tolerance(1.e-10),
      update_ale(true),
      update_fluid_velocity(true),
      update_fluid_pressure(true)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "FSI")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("AccelerationMethod",
                        acceleration_method,
                        "Acceleration method.",
                        Patterns::Enum<AccelerationMethod>(),
                        true);
      prm.add_parameter(
        "AbsTol", abs_tol, "Absolute solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter(
        "RelTol", rel_tol, "Relative solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter("OmegaInit",
                        omega_init,
                        "Initial relaxation parameter.",
                        dealii::Patterns::Double(0.0, 1.0),
                        true);
      prm.add_parameter("InitialGuessCouplingScheme",
                        initial_guess_coupling_scheme,
                        "Scheme for initial guess for the FSI coupling loop at every time step.",
                        Patterns::Enum<InitialGuessCouplingScheme>(),
                        false);
      prm.add_parameter("ReusedTimeSteps",
                        reused_time_steps,
                        "Number of time steps reused for acceleration.",
                        dealii::Patterns::Integer(0, 100),
                        false);
      prm.add_parameter("PartitionedIterMax",
                        partitioned_iter_max,
                        "Maximum number of fixed-point iterations.",
                        dealii::Patterns::Integer(1, 1000),
                        true);
      prm.add_parameter("GeometricTolerance",
                        geometric_tolerance,
                        "Tolerance used to locate points at FSI interface.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("UpdateALE",
                        update_ale,
                        "Include ALE update in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("UpdateFluidVelocity",
                        update_fluid_velocity,
                        "Include fluid velocity subproblem in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("UpdateFluidPressure",
                        update_fluid_pressure,
                        "Include fluid pressure subproblem in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
    }
    prm.leave_subsection();
  }

  AccelerationMethod         acceleration_method;
  double                     abs_tol;
  double                     rel_tol;
  double                     omega_init;
  InitialGuessCouplingScheme initial_guess_coupling_scheme;
  unsigned int               reused_time_steps;
  unsigned int               partitioned_iter_max;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;

  // Semi-implicit coupling variants iteratively update only a subset of all
  // fields after the first coupling step. Stability properties are altered,
  // but temporal order of accuracy is preserved starting from suitable
  // extrapolations in time controlled via `InitialGuessCouplingScheme`.
  bool update_ale;
  bool update_fluid_velocity;
  bool update_fluid_pressure;
};

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_ */
