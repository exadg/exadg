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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_

namespace ExaDG
{
namespace IncNS
{
enum class InitializeSolutionWith
{
  ZeroFunction,
  AnalyticalSolution
};

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const lambda) : dealii::Function<dim>(dim, 0.0), lambda(lambda)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const pi = dealii::numbers::PI;

    double result = 0.0;
    if(component == 0)
      result = 1.0 - std::exp(lambda * p[0]) * std::cos(2 * pi * p[1]);
    else if(component == 1)
      result = lambda / 2.0 / pi * std::exp(lambda * p[0]) * std::sin(2 * pi * p[1]);

    return result;
  }

private:
  double const lambda;
};

template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const lambda)
    : dealii::Function<dim>(1 /*n_components*/, 0.0), lambda(lambda)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double const result = 0.5 * (1.0 - std::exp(2.0 * lambda * p[0]));

    return result;
  }

private:
  double const lambda;
};

template<int dim>
class NeumannBoundaryVelocity : public dealii::Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const & formulation_viscous, double const lambda)
    : dealii::Function<dim>(dim, 0.0), formulation_viscous(formulation_viscous), lambda(lambda)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const pi = dealii::numbers::PI;

    double result = 0.0;
    if(formulation_viscous == FormulationViscousTerm::LaplaceFormulation)
    {
      if(component == 0)
        result = -lambda * std::exp(lambda) * std::cos(2 * pi * p[1]);
      else if(component == 1)
        result = std::pow(lambda, 2.0) / 2 / pi * std::exp(lambda) * std::sin(2 * pi * p[1]);
    }
    else if(formulation_viscous == FormulationViscousTerm::DivergenceFormulation)
    {
      if(component == 0)
        result = -2.0 * lambda * std::exp(lambda) * std::cos(2 * pi * p[1]);
      else if(component == 1)
        result =
          (std::pow(lambda, 2.0) / 2 / pi + 2.0 * pi) * std::exp(lambda) * std::sin(2 * pi * p[1]);
    }
    else
    {
      AssertThrow(formulation_viscous == FormulationViscousTerm::LaplaceFormulation ||
                    formulation_viscous == FormulationViscousTerm::DivergenceFormulation,
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }

private:
  FormulationViscousTerm const formulation_viscous;
  double const                 lambda;
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
    this->param.formulation_viscous_term    = formulation_viscous;
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
    this->param.max_velocity                  = 3.6;
    this->param.cfl                           = 1.0e-2;
    this->param.time_step_size                = 1.0e-3;
    this->param.order_time_integrator         = 3;    // 1; // 2; // 3;
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

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6, 100);
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
    this->param.nonlinear_solver_data_momentum =
      NonlinearSolver::SolverData(100, 1.e-20, 1.e-6, NonlinearSolver::SolverType::Newton);

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
    this->param.nonlinear_solver_data_coupled =
      NonlinearSolver::SolverData(100, 1.0e-12, 1.0e-6, NonlinearSolver::SolverType::Newton);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 1000);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // Jacobi; //Chebyshev; //GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
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
    double const left = -1.0, right = 1.0;
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    // set boundary indicator
    for(auto cell : this->grid->triangulation->cell_iterators())
    {
      for(auto const & f : cell->face_indices())
      {
        if((std::fabs(cell->face(f)->center()(0) - right) < 1e-12))
          cell->face(f)->set_boundary_id(1);
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
      pair(0, new AnalyticalSolutionVelocity<dim>(lambda)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new NeumannBoundaryVelocity<dim>(formulation_viscous, lambda)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionPressure<dim>(lambda)));
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<dealii::Function<dim>> initial_solution_velocity;
    std::shared_ptr<dealii::Function<dim>> initial_solution_pressure;
    if(initialize_solution_with == InitializeSolutionWith::ZeroFunction)
    {
      initial_solution_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
      initial_solution_pressure.reset(new dealii::Functions::ZeroFunction<dim>(1));
    }
    else if(initialize_solution_with == InitializeSolutionWith::AnalyticalSolution)
    {
      initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(lambda));
      initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(lambda));
    }

    this->field_functions->initial_solution_velocity = initial_solution_velocity;
    this->field_functions->initial_solution_pressure = initial_solution_pressure;
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(lambda));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory        = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename         = this->output_parameters.filename;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.degree           = this->param.degree_u;

    // calculation of velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(lambda));
    pp_data.error_data_u.name = "velocity";

    // ... pressure error
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(lambda));
    pp_data.error_data_p.name = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  InitializeSolutionWith const initialize_solution_with =
    InitializeSolutionWith::AnalyticalSolution;

  FormulationViscousTerm const formulation_viscous = FormulationViscousTerm::LaplaceFormulation;

  double const viscosity = 2.5e-2;
  double const lambda =
    0.5 / viscosity -
    std::pow(0.25 / std::pow(viscosity, 2.0) + 4.0 * std::pow(dealii::numbers::PI, 2.0), 0.5);

  double const start_time = 0.0;
  double const end_time   = 1.0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_ */
