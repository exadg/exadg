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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const L = 1.0;

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Steady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = 1.0e-2;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Steady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Implicit;
    this->param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    this->param.adaptive_time_stepping          = false;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = 1.0;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    // Explicit: CFL_crit = 0.35 (0.4 unstable), ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable)
    this->param.cfl_oif               = 3.0;
    this->param.cfl                   = this->param.cfl_oif * 1.0;
    this->param.time_step_size        = 1.0e-1;
    this->param.order_time_integrator = 2;
    this->param.start_with_low_order  = true;

    // pseudo-timestepping for steady-state problems
    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-8;

    // restart
    this->param.restart_data.write_restart       = false;
    this->param.restart_data.interval_time       = 5.0;
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;
    this->param.restart_data.filename            = "output/cavity/cavity_restart";

    // output of solver information
    this->param.solver_info_data.interval_time_steps = 1;
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.apply_penalty_terms_in_postprocessing_step = false;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;
    this->param.solver_data_block_diagonal = SolverData(1000, 1.e-12, 1.e-2, 1000);
    this->param.quad_rule_linearization    = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8, 100);
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

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
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    this->param.solver_momentum = SolverMomentum::GMRES; // FGMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
    else
      this->param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = true;
    this->param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    this->param.multigrid_data_momentum.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_momentum.smoother_data.iterations        = 5;
    this->param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-8);

    // linear solver
    this->param.solver_coupled = SolverCoupled::FGMRES; // FGMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 1000);
    else
      this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 1000);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block =
      MultigridOperatorType::ReactionConvectionDiffusion;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;

    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    this->param.exact_inversion_of_velocity_block = false; // true;
    this->param.solver_data_velocity_block        = SolverData(1e4, 1.e-12, 1.e-6, 100);

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    this->param.multigrid_data_pressure_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.exact_inversion_of_laplace_operator = false;
    this->param.solver_data_pressure_block          = SolverData(1e4, 1.e-12, 1.e-6, 100);
  }

  void
  create_grid() final
  {
    double const left = 0.0, right = L;
    GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    // set boundary indicator
    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if((std::fabs(cell->face(face)->center()(1) - L) < 1e-12))
          cell->face(face)->set_boundary_id(1);
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    // all boundaries have ID = 0 by default -> Dirichlet boundaries

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(velocity)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));
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

    // write output for visualization of results
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.directory            = this->output_directory + "vtu/";
    pp_data.output_data.filename             = this->output_name;
    pp_data.output_data.start_time           = start_time;
    pp_data.output_data.interval_time        = (end_time - start_time) / 100;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.write_vorticity      = true;
    pp_data.output_data.write_streamfunction = true; // false;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.degree               = this->param.degree_u;

    // consider line plots only for two-dimensional case
    if(dim == 2)
    {
      // line plot data
      pp_data.line_plot_data.calculate           = false;
      pp_data.line_plot_data.line_data.directory = this->output_directory;

      // which quantities
      std::shared_ptr<Quantity> quantity_u;
      quantity_u.reset(new Quantity());
      quantity_u->type = QuantityType::Velocity;
      std::shared_ptr<Quantity> quantity_p;
      quantity_p.reset(new Quantity());
      quantity_p->type = QuantityType::Pressure;

      // lines
      std::shared_ptr<Line<dim>> vert_line, hor_line;

      // vertical line
      vert_line.reset(new Line<dim>());
      vert_line->begin    = Point<dim>(0.5, 0.0);
      vert_line->end      = Point<dim>(0.5, 1.0);
      vert_line->name     = this->output_name + "_vert_line";
      vert_line->n_points = 100001; // 2001;
      vert_line->quantities.push_back(quantity_u);
      vert_line->quantities.push_back(quantity_p);
      pp_data.line_plot_data.line_data.lines.push_back(vert_line);

      // horizontal line
      hor_line.reset(new Line<dim>());
      hor_line->begin    = Point<dim>(0.0, 0.5);
      hor_line->end      = Point<dim>(1.0, 0.5);
      hor_line->name     = this->output_name + "_hor_line";
      hor_line->n_points = 10001; // 2001;
      hor_line->quantities.push_back(quantity_u);
      hor_line->quantities.push_back(quantity_p);
      pp_data.line_plot_data.line_data.lines.push_back(hor_line);
    }

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
