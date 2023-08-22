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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_

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

enum class UpdateMethod
{
  Undefined,
  Implicit,
  GeometricExplicit,
  ImplicitPressureStructure,
  ImplicitVelocityStructure
};

enum class CouplingMethod
{
  Undefined,
  DirichletNeumann,
  DirichletRobinFixedParameter,
  DirichletRobinAdaptiveParameter
};

struct Parameters
{
  Parameters()
    : acceleration_method(AccelerationMethod::Undefined),
      update_method(UpdateMethod::Implicit),
	  coupling_method(CouplingMethod::DirichletNeumann),
	  robin_parameter_scale(0.0),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      use_extrapolation(true),
      reused_time_steps(0),
      partitioned_iter_max(100),
      geometric_tolerance(1.e-10)
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
        "UpdateMethod", update_method, "Update method.", Patterns::Enum<UpdateMethod>(), false);
      prm.add_parameter("CouplingMethod", coupling_method, "Coupling method.", Patterns::Enum<CouplingMethod>(), false);
      prm.add_parameter("RobinParameterScale", robin_parameter_scale, "Parameter scale for Robin coupling.", dealii::Patterns::Double(), false);
      prm.add_parameter(
        "AbsTol", abs_tol, "Absolute solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter(
        "RelTol", rel_tol, "Relative solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter("OmegaInit",
                        omega_init,
                        "Initial relaxation parameter.",
                        dealii::Patterns::Double(0.0, 1.0),
                        true);
      prm.add_parameter("UseExtrapolation",
                        use_extrapolation,
                        "Extrapolate for initial guess in coupling scheme.",
                        dealii::Patterns::Bool(),
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
    }
    prm.leave_subsection();
  }

  AccelerationMethod acceleration_method;
  UpdateMethod       update_method;
  CouplingMethod     coupling_method;
  double             robin_parameter_scale;
  double             abs_tol;
  double             rel_tol;
  double             omega_init;
  bool               use_extrapolation;
  unsigned int       reused_time_steps;
  unsigned int       partitioned_iter_max;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;
};
} // namespace FSI
} // namespace ExaDG



#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_ */
