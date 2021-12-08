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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_

// ExaDG
#include <exadg/grid/mesh_movement_functions.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class AnalyticalSolutionVelocity : public Functions::ConstantFunction<dim>
{
public:
  AnalyticalSolutionVelocity() : Functions::ConstantFunction<dim>(1.0, dim)
  {
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Functions::ConstantFunction<dim>
{
public:
  AnalyticalSolutionPressure() : Functions::ConstantFunction<dim>(1.0, 1)
  {
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
      prm.add_parameter("ALE",
                        ALE,
                        "Moving mesh (ALE).",
                        Patterns::Bool());
    prm.leave_subsection();
    // clang-format on
  }

  double const left  = -0.5;
  double const right = 0.5;

  double const start_time = 0.0;
  double const end_time   = 10.0;

  bool ALE = true;

  void
  set_input_parameters(unsigned int const degree) final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side             = false;

    // ALE
    this->param.ale_formulation = ALE;
    this->param.neumann_with_variable_normal_vector =
      false; // no Neumann boundaries for this test case

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = 2.5e-2;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = false;
    this->param.adaptive_time_stepping          = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.time_step_size                  = 0.25;
    this->param.max_velocity                    = 1.0;
    this->param.cfl                             = 0.25;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.c_eff                           = 8.0;
    this->param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    this->param.cfl_oif                         = this->param.cfl / 1.0;

    // output of solver information
    this->param.solver_info_data.interval_time = this->param.end_time - this->param.start_time;

    // SPATIAL DISCRETIZATION
    this->param.triangulation_type = TriangulationType::Distributed;
    this->param.degree_u           = degree;
    this->param.degree_p           = DegreePressure::MixedOrder;
    this->param.mapping            = MappingType::Isoparametric;

    // convective term
    this->param.upwind_factor = 1.0;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Strong;   // TODO //Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Strong; // TODO //Weak;

    // pressure level is undefined
    this->param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalSolutionInPoint;

    // div-div and continuity penalty
    this->param.use_divergence_penalty               = true;
    this->param.divergence_penalty_factor            = 1.0e0;
    this->param.use_continuity_penalty               = true;
    this->param.continuity_penalty_factor            = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components        = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data = true;
    if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      this->param.apply_penalty_terms_in_postprocessing_step = false;
    else
      this->param.apply_penalty_terms_in_postprocessing_step = true;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;
    this->param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS
    this->param.store_previous_boundary_values = true;

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-14, 1.e-14, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-14, 1.e-14);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
    this->param.preconditioner_block_diagonal_projection =
      Elementwise::Preconditioner::InverseMassMatrix;
    this->param.solver_data_block_diagonal_projection = SolverData(1000, 1.e-12, 1.e-2, 1000);

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;
    this->param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

    // viscous step
    this->param.solver_viscous                                = SolverViscous::CG;
    this->param.solver_data_viscous                           = SolverData(1000, 1.e-14, 1.e-14);
    this->param.preconditioner_viscous                        = PreconditionerViscous::Multigrid;
    this->param.multigrid_data_viscous.type                   = MultigridType::hMG;
    this->param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.update_preconditioner_viscous                 = false;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation =
      std::min(2, (int)this->param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    this->param.rotational_formulation = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-12);

    // linear solver
    this->param.solver_momentum                  = SolverMomentum::FGMRES;
    this->param.solver_data_momentum             = SolverData(1e4, 1.e-14, 1.e-14, 100);
    this->param.update_preconditioner_momentum   = false;
    this->param.preconditioner_momentum          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.multigrid_data_momentum.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;

    // COUPLED NAVIER-STOKES SOLVER
    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-12);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-14, 100);

    // preconditioner linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = false;

    // preconditioner momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.type     = MultigridType::phMG;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    // coarse grid solver
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev; // GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }


  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    GridGenerator::hyper_cube(*grid->triangulation, left, right);

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }

  std::shared_ptr<Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal                       = MeshMovementAdvanceInTime::Sin;
    data.shape                          = MeshMovementShape::Sin;
    data.dimensions[0]                  = std::abs(right - left);
    data.dimensions[1]                  = std::abs(right - left);
    data.amplitude                      = 0.08 * (right - left);
    data.period                         = (end_time - start_time) / 10.0;
    data.t_start                        = start_time;
    data.t_end                          = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_conditions() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>()));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    this->field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>());
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 100;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_ */
