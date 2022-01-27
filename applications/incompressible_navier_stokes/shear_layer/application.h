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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_

/*
 * For a description of the test case, see
 *
 * Brown and Minion (J. Comput. Phys. 122 (1995), 165-183)
 */

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity(double const rho, double const delta)
    : dealii::Function<dim>(dim, 0.0), rho(rho), delta(delta)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;
    if(component == 0)
      result = std::tanh(rho * (0.25 - std::abs(0.5 - p[1])));
    else if(component == 1)
      result = delta * std::sin(2.0 * dealii::numbers::PI * p[0]);

    return result;
  }

private:
  double const rho, delta;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  bool const   inviscid  = true;
  double const viscosity = inviscid ? 0.0 : 1.0e-4; // Re = 10^4
  double const rho       = 30.0;
  double const delta     = 0.05;

  double const start_time = 0.0;
  double const end_time   = 4.0;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = ProblemType::Unsteady;
    if(inviscid)
      this->param.equation_type = EquationType::Euler;
    else
      this->param.equation_type = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping        = true;
    this->param.adaptive_time_stepping_limiting_factor = 3.0;
    this->param.max_velocity                           = 1.5;
    this->param.cfl                                    = 0.25;
    this->param.cfl_exponent_fe_degree_velocity        = 1.5;
    this->param.time_step_size                         = 1.0e-4;
    this->param.order_time_integrator                  = 2;
    this->param.start_with_low_order                   = true;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 40;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity-pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // penalty terms
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // formulation
    this->param.store_previous_boundary_values = false;

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  }

  void
  create_grid() final
  {
    double const left = 0.0, right = 1.0;
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    // use periodic boundary conditions
    // x-direction
    this->grid->triangulation->begin()->face(0)->set_all_boundary_ids(0);
    this->grid->triangulation->begin()->face(1)->set_all_boundary_ids(1);
    // y-direction
    this->grid->triangulation->begin()->face(2)->set_all_boundary_ids(2);
    this->grid->triangulation->begin()->face(3)->set_all_boundary_ids(3);

    dealii::GridTools::collect_periodic_faces(
      *this->grid->triangulation, 0, 1, 0, this->grid->periodic_faces);
    dealii::GridTools::collect_periodic_faces(
      *this->grid->triangulation, 2, 3, 1, this->grid->periodic_faces);
    this->grid->triangulation->add_periodicity(this->grid->periodic_faces);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(rho, delta));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output     = this->write_output;
    pp_data.output_data.directory        = this->output_directory + "vtu/";
    pp_data.output_data.filename         = this->output_name;
    pp_data.output_data.start_time       = start_time;
    pp_data.output_data.interval_time    = (end_time - start_time) / 40;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.write_vorticity  = true;
    pp_data.output_data.degree           = this->param.degree_u;

    // kinetic energy
    pp_data.kinetic_energy_data.calculate                  = true;
    pp_data.kinetic_energy_data.evaluate_individual_terms  = false;
    pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
    pp_data.kinetic_energy_data.viscosity                  = viscosity;
    pp_data.kinetic_energy_data.directory                  = this->output_directory;
    pp_data.kinetic_energy_data.filename                   = this->output_name;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_ */
