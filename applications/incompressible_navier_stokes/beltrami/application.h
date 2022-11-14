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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const viscosity)
    : dealii::Function<dim>(dim, 0.0), nu(viscosity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t = this->get_time();
    double const a = 0.25 * dealii::numbers::PI;
    double const d = 2 * a;

    double result = 0.0;
    // clang-format off
    if (component == 0)
      result = -a*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-nu*d*d*t);
    else if (component == 1)
      result = -a*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-nu*d*d*t);
    else if (component == 2)
      result = -a*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-nu*d*d*t);
    // clang-format on

    return result;
  }

private:
  double const nu;
};

template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const viscosity)
    : dealii::Function<dim>(1 /*n_components*/, 0.0), nu(viscosity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double const t = this->get_time();
    double const a = 0.25 * dealii::numbers::PI;
    double const d = 2 * a;

    // clang-format off
    double result = -a*a*0.5*(std::exp(2*a*p[0]) + std::exp(2*a*p[1]) + std::exp(2*a*p[2]) +
                     2*std::sin(a*p[0]+d*p[1])*std::cos(a*p[2]+d*p[0])*std::exp(a*(p[1]+p[2])) +
                     2*std::sin(a*p[1]+d*p[2])*std::cos(a*p[0]+d*p[1])*std::exp(a*(p[2]+p[0])) +
                     2*std::sin(a*p[2]+d*p[0])*std::cos(a*p[1]+d*p[2])*std::exp(a*(p[0]+p[1]))) * std::exp(-2*nu*d*d*t);
    // clang-format on

    return result;
  }

private:
  double const nu;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type = SolverType::Unsteady;
    this->param.temporal_discretization =
      TemporalDiscretization::BDFPressureCorrection; // BDFCoupledSolution;
    this->param.treatment_of_convective_term =
      TreatmentOfConvectiveTerm::Explicit; // Explicit; //Implicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.max_velocity                  = 3.5;
    this->param.cfl                           = 1.0e-1;
    this->param.time_step_size                = 1.0e-3; // 1.0e-4;
    this->param.order_time_integrator         = 2;      // 1; // 2; // 3;
    this->param.start_with_low_order          = false;  // true; // false;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // pressure level is undefined
    this->param.adjust_pressure_level =
      AdjustPressureLevel::ApplyAnalyticalMeanValue; // ApplyAnalyticalSolutionInPoint;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8, 100);
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    this->param.preconditioner_viscous = PreconditionerViscous::Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-10, 1.e-8);

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = false;

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER
    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-8);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::GMRES;
    this->param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion; // CahouetChabard;
  }

  void
  create_grid() final
  {
    double const left = -1.0, right = 1.0;
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(viscosity)));

    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(viscosity));
    this->field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.output_data.directory        = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename         = this->output_parameters.filename;
    pp_data.output_data.write_divergence = false;
    pp_data.output_data.degree           = this->param.degree_u;

    // calculation of velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(viscosity));
    pp_data.error_data_u.name = "velocity";

    // ... pressure error
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(viscosity));
    pp_data.error_data_p.name = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const viscosity = 0.1;

  double const start_time = 0.0;
  double const end_time   = 1.0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_ */
