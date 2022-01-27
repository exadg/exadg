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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const H, double const max_velocity)
    : dealii::Function<dim>(dim, 0.0), H(H), max_velocity(max_velocity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = max_velocity * (p[1] + H / 2.) / H;

    return result;
  }

private:
  double const H, max_velocity;
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

  double const H            = 1.0;
  double const L            = 4.0;
  double const max_velocity = 1.0;

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type             = ProblemType::Steady; // Unsteady;
    this->param.equation_type            = EquationType::NavierStokes;
    this->param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    this->param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = 1.0;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type = SolverType::Steady; // Unsteady;
    this->param.temporal_discretization =
      TemporalDiscretization::BDFCoupledSolution; // BDFDualSplittingScheme;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit; // Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.max_velocity                  = max_velocity;
    this->param.cfl                           = 1.0e-1;
    this->param.time_step_size                = 1.0e-1;
    this->param.order_time_integrator         = 2;    // 1; // 2; // 3;
    this->param.start_with_low_order          = true; // true; // false;

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

    // divergence and continuity penalty terms
    this->param.apply_penalty_terms_in_postprocessing_step = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6);
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
    this->param.solver_data_viscous    = SolverData(1000, 1.e-20, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-4, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = false;

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-2, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.type     = MultigridType::hMG;
    this->param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid() final
  {
    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 2;
    dealii::Point<dim> point1, point2;
    point1[0] = 0.0;
    point1[1] = -H / 2;
    if(dim == 3)
      point1[2] = 0.0;

    point2[0] = L;
    point2[1] = H / 2;
    if(dim == 3)
      point2[2] = H;

    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      point1,
                                                      point2);

    // set boundary indicator
    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if((std::fabs(cell->face(face)->center()(0) - L) < 1e-12))
          cell->face(face)->set_boundary_id(1);
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(H, max_velocity)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
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
    pp_data.output_data.interval_time    = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.degree           = this->param.degree_u;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(H, max_velocity));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time) / 20;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time) / 20;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_ */
