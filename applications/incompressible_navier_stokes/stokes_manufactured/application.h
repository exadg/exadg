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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_MANUFACTURED_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_MANUFACTURED_H_

namespace ExaDG
{
namespace IncNS
{
// perform stability analysis and compute eigenvalue spectrum
// For this analysis one has to use the BDF1 scheme and homogeneous boundary conditions!!!
bool const STABILITY_ANALYSIS = false;

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const nu) : dealii::Function<dim>(dim, 0.0), viscosity(nu)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double t      = this->get_time();
    double result = 0.0;

    double const a      = 2.883356;
    double const lambda = viscosity * (1. + a * a);

    double exp_t  = std::exp(-lambda * t);
    double sin_x  = std::sin(p[0]);
    double cos_x  = std::cos(p[0]);
    double cos_a  = std::cos(a);
    double sin_ay = std::sin(a * p[1]);
    double cos_ay = std::cos(a * p[1]);
    double sinh_y = std::sinh(p[1]);
    double cosh_y = std::cosh(p[1]);
    if(component == 0)
      result = exp_t * sin_x * (a * sin_ay - cos_a * sinh_y);
    else if(component == 1)
      result = exp_t * cos_x * (cos_ay + cos_a * cosh_y);

    if(STABILITY_ANALYSIS == true)
      result = 0;

    return result;
  }

private:
  double const viscosity;
};


template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const nu) : dealii::Function<dim>(1, 0.0), viscosity(nu)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double t      = this->get_time();
    double result = 0.0;

    double const a      = 2.883356;
    double const lambda = viscosity * (1. + a * a);

    double exp_t  = std::exp(-lambda * t);
    double cos_x  = std::cos(p[0]);
    double cos_a  = std::cos(a);
    double sinh_y = std::sinh(p[1]);
    result        = lambda * cos_a * cos_x * sinh_y * exp_t;

    if(STABILITY_ANALYSIS == true)
      result = 0;

    return result;
  }

private:
  double const viscosity;
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
    this->param.problem_type             = ProblemType::Unsteady;
    this->param.equation_type            = EquationType::Stokes;
    this->param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    this->param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = this->param.end_time;
    this->param.order_time_integrator         = 1; // 1; // 2; // 3;
    if(STABILITY_ANALYSIS)
      this->param.order_time_integrator = 1;
    this->param.start_with_low_order = true; // true; // false;

    // output of solver information
    this->param.solver_info_data.interval_time =
      1.0; //(this->param.end_time-this->param.start_time)/10;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // gradient term
    this->param.gradp_integrated_by_parts = true;
    this->param.gradp_use_boundary_data   = true;
    this->param.gradp_formulation         = FormulationPressureGradientTerm::Weak;

    // divergence term
    this->param.divu_integrated_by_parts = true;
    this->param.divu_use_boundary_data   = true;
    this->param.divu_formulation         = FormulationVelocityDivergenceTerm::Weak;

    // pressure level is undefined
    this->param.adjust_pressure_level =
      AdjustPressureLevel::ApplyZeroMeanValue; // ApplyAnalyticalSolutionInPoint;

    // div-div and continuity penalty terms
    this->param.use_divergence_penalty                     = true;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8);

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-8);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;


    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = false;

    // formulation
    this->param.order_pressure_extrapolation = this->param.order_time_integrator - 1;
    this->param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)

    // linear solver
    this->param.solver_coupled      = SolverCoupled::GMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    (void)mapping;
    (void)multigrid_mappings;

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        double const left = -1.0, right = 1.0;
        dealii::GridGenerator::hyper_cube(tria, left, right);

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    // test case with pure Dirichlet boundary conditions for velocity
    // all boundaries have ID = 0 by default

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(viscosity)));

    // fill boundary descriptor pressure
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
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.output_data.directory        = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename         = this->output_parameters.filename;
    pp_data.output_data.write_divergence = false;
    pp_data.output_data.degree           = this->param.degree_u;

    // calculation of velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(viscosity));
    pp_data.error_data_u.calculate_relative_errors = true; // false;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(viscosity));
    pp_data.error_data_p.calculate_relative_errors = true; // false;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const viscosity = 1.0e0;

  double const start_time = 0.0;
  double const end_time   = 1.0e-1;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_MANUFACTURED_H_ */
