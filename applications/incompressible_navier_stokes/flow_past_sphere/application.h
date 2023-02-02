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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_SPHERE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_SPHERE_H_

#include "include/grid.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;
using namespace FlowPastSphere;

template<int dim>
class InflowBC : public Function<dim>
{
public:
  InflowBC() : Function<dim>(dim, 0.0)
  {
  }

  virtual double
  value(Point<dim> const &, unsigned int const component = 0) const final
  {
    if(component == 0)
      return 1.0;
    else
      return 0.0;
  }
};


template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("CFL",        cfl_number, "CFL number.",      Patterns::Double(0.0, 1.0e6), true);
    prm.add_parameter("Viscosity",  viscosity,  "Fluid viscosity.", Patterns::Double(0.0, 1e30),  false);
    prm.leave_subsection();
    // clang-format on
  }

  double viscosity = 1.e-3;

  double cfl_number = 1.0;

  // start and end time
  // use a large value for test_case = 1 (steady problem)
  // in order to not stop pseudo-timestepping approach before having converged
  double const start_time = 0.0;
  double const end_time   = 100.;

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-4;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

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
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = 1.;
    this->param.cfl                             = cfl_number;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-3;
    this->param.time_step_size_max              = 1.0e-2;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 20.0;

    // pseudo-timestepping for steady-state problems
    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-8;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // divergence penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 5.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;
    this->param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 30);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.p_sequence            = PSequenceType::Bisect;
    this->param.multigrid_data_pressure_poisson.use_global_coarsening = true;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.iterations = 4;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::AMG;
    this->param.update_preconditioner_pressure_poisson = false;

    // projection step
    this->param.solver_projection                = SolverProjection::CG;
    this->param.solver_data_projection           = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_projection        = PreconditionerProjection::InverseMassMatrix;
    this->param.update_preconditioner_projection = false;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous                = SolverViscous::CG;
    this->param.solver_data_viscous           = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_viscous        = PreconditionerViscous::InverseMassMatrix;
    this->param.update_preconditioner_viscous = false;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_momentum      = SolverMomentum::FGMRES;
    this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);

    this->param.update_preconditioner_momentum                   = true;
    this->param.update_preconditioner_momentum_every_newton_iter = 10;
    this->param.update_preconditioner_momentum_every_time_steps  = 10;

    this->param.preconditioner_momentum = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_momentum =
      MultigridOperatorType::ReactionConvectionDiffusion;
    this->param.multigrid_data_momentum.type                   = MultigridType::phMG;
    this->param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    this->param.multigrid_data_momentum.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_momentum.smoother_data.iterations        = 1;
    this->param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
    this->param.multigrid_data_momentum.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::BlockJacobi;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);

    this->param.update_preconditioner_coupled                   = true;
    this->param.update_preconditioner_coupled_every_newton_iter = 10;
    this->param.update_preconditioner_coupled_every_time_steps  = 10;

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block =
      MultigridOperatorType::ReactionConvectionDiffusion;
    this->param.multigrid_data_velocity_block.type                   = MultigridType::phMG;
    this->param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 1;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;
    this->param.multigrid_data_velocity_block.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::BlockJacobi;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    this->param.multigrid_data_pressure_block.type = MultigridType::cphMG;
  }


  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> & tria,
          unsigned int const                global_refinements,
          std::vector<unsigned int> const & vector_local_refinements) {
        (void)vector_local_refinements;
        create_sphere_grid<dim>(tria, global_refinements);
      };

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
                                                              this->param.grid,
                                                              this->param.use_global_coarsening(),
                                                              lambda_create_triangulation);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(pair(1, new InflowBC<dim>()));
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(3, new Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->symmetry_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
    this->boundary_descriptor->pressure->neumann_bc.insert(3);
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::string name = this->output_parameters.filename + "_l" +
                       std::to_string(this->param.grid.n_refine_global) + "_k" +
                       std::to_string(this->param.degree_u);

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 50.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = name;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.write_vorticity    = true;
    pp_data.output_data.write_q_criterion  = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_surface_mesh = false;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_grid         = false;
    pp_data.output_data.degree             = this->param.degree_u;

    // lift and drag
    pp_data.lift_and_drag_data.time_control_data.is_active                = true;
    pp_data.lift_and_drag_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.lift_and_drag_data.time_control_data.start_time               = start_time;
    pp_data.lift_and_drag_data.viscosity                                  = viscosity;

    pp_data.lift_and_drag_data.reference_value = 1.0 / 2.0;

    // surface for calculation of lift and drag coefficients has boundary_ID = 2
    pp_data.lift_and_drag_data.boundary_IDs.insert(3);

    pp_data.lift_and_drag_data.directory     = this->output_parameters.directory;
    pp_data.lift_and_drag_data.filename_lift = name + "_lift";
    pp_data.lift_and_drag_data.filename_drag = name + "_drag";

    // pressure difference
    pp_data.pressure_difference_data.time_control_data.is_active                = true;
    pp_data.pressure_difference_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.pressure_difference_data.time_control_data.start_time               = start_time;
    Point<dim> point_1, point_2;
    point_1[0]                               = -radius;
    point_2[0]                               = radius;
    pp_data.pressure_difference_data.point_1 = point_1;
    pp_data.pressure_difference_data.point_2 = point_2;

    pp_data.pressure_difference_data.directory = this->output_parameters.directory;
    pp_data.pressure_difference_data.filename  = name + "_pressure_difference";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
