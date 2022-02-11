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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

// flow past cylinder application
#include "include/grid.h"

namespace ExaDG
{
namespace IncNS
{
using namespace FlowPastCylinder;

template<int dim>
class InflowBC : public dealii::Function<dim>
{
public:
  InflowBC(double const                                        Um,
           double const                                        H,
           double const                                        end_time,
           unsigned int const                                  test_case,
           bool const                                          use_random_perturbations,
           std::vector<double> const &                         y,
           std::vector<double> const &                         z,
           std::vector<dealii::Tensor<1, dim, double>> const & u)
    : dealii::Function<dim>(dim, 0.0),
      Um(Um),
      H(H),
      end_time(end_time),
      test_case(test_case),
      use_random_perturbations(use_random_perturbations),
      y_vector(y),
      z_vector(z),
      u_vector(u)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(component == 0)
    {
      double const pi          = dealii::numbers::PI;
      double const T           = 1.0;
      double       coefficient = dealii::Utilities::fixed_power<dim - 1>(4.) * Um /
                           dealii::Utilities::fixed_power<2 * dim - 2>(H);

      if(test_case == 1)
        result = coefficient * x[1] * (H - x[1]);
      else if(test_case == 2)
        result =
          coefficient * x[1] * (H - x[1]) * ((t / T) < 1.0 ? std::sin(pi / 2. * t / T) : 1.0);
      else if(test_case == 3)
        result = coefficient * x[1] * (H - x[1]) * std::sin(pi * t / end_time);
      else
        AssertThrow(false, dealii::ExcMessage("Not implemented."));

      if(dim == 3)
        result *= x[2] * (H - x[2]);

      if(use_random_perturbations)
      {
        double perturbation = 0.0;

        if(dim == 2)
          perturbation = linear_interpolation_1d(x[1], y_vector, u_vector, component);
        else if(dim == 3)
        {
          dealii::Point<dim> point_3d;
          point_3d[0] = x[0];
          point_3d[1] = x[1];
          point_3d[2] = x[2];

          perturbation =
            linear_interpolation_2d_cartesian(point_3d, y_vector, z_vector, u_vector, component);
        }
        else
          AssertThrow(false, dealii::ExcMessage("Not implemented."));

        result += perturbation;
      }
    }

    return result;
  }

private:
  double const       Um, H, end_time;
  unsigned int const test_case;

  // perturbations
  bool const                                  use_random_perturbations;
  std::vector<double> const &                 y_vector;
  std::vector<double> const &                 z_vector;
  std::vector<dealii::Tensor<1, dim, double>> u_vector;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    if(use_perturbation)
    {
      initialize_y_and_z_values();
      initialize_velocity_values();
    }
  }

  void
  initialize_y_and_z_values()
  {
    AssertThrow(n_points_y >= 2, dealii::ExcMessage("Variable n_points_y is invalid"));
    if(dim == 3)
      AssertThrow(n_points_z >= 2, dealii::ExcMessage("Variable n_points_z is invalid"));

    // 0 <= y <= H
    for(unsigned int i = 0; i < n_points_y; ++i)
      y_values[i] = double(i) / double(n_points_y - 1) * H;

    // 0 <= z <= H
    if(dim == 3)
      for(unsigned int i = 0; i < n_points_z; ++i)
        z_values[i] = double(i) / double(n_points_z - 1) * H;
  }

  void
  initialize_velocity_values()
  {
    AssertThrow(n_points_y >= 2, dealii::ExcMessage("Variable n_points_y is invalid"));
    if(dim == 3)
      AssertThrow(n_points_z >= 2, dealii::ExcMessage("Variable n_points_z is invalid"));

    for(unsigned int iy = 0; iy < n_points_y; ++iy)
    {
      for(unsigned int iz = 0; iz < n_points_z; ++iz)
      {
        dealii::Tensor<1, dim, double> velocity;

        if(use_perturbation == true)
        {
          // Add random perturbation
          double const y           = y_values[iy];
          double const z           = z_values[iz];
          double       coefficient = dealii::Utilities::fixed_power<dim - 1>(4.) * Um /
                               dealii::Utilities::fixed_power<2 * dim - 2>(H);
          double perturbation =
            amplitude_perturbation * coefficient * ((double)rand() / RAND_MAX - 0.5) / 0.5;
          perturbation *= y * (H - y);
          if(dim == 3)
            perturbation *= z * (H - z);

          velocity[0] += perturbation;
        }

        velocity_values[iy * n_points_z + iz] = velocity;
      }
    }
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("TestCase",     test_case,            "Number of test case.", dealii::Patterns::Integer(1,3));
      prm.add_parameter("CylinderType", cylinder_type_string, "Type of cylinder.",    dealii::Patterns::Selection("circular|square"));
      prm.add_parameter("CFL",          cfl_number,           "CFL number.",          dealii::Patterns::Double(0.0, 1.0e6), true);
    prm.leave_subsection();
    // clang-format on
  }

  // string to read parameter
  std::string cylinder_type_string = "circular";

  // select test case according to Schaefer and Turek benchmark definition: 2D-1/2/3, 3D-1/2/3
  unsigned int test_case = 3; // 1, 2 or 3

  ProblemType  problem_type = ProblemType::Unsteady;
  double const Um = (dim == 2 ? (test_case == 1 ? 0.3 : 1.5) : (test_case == 1 ? 0.45 : 2.25));

  double const viscosity = 1.e-3;

  double cfl_number = 1.0;

  // start and end time
  // use a large value for test_case = 1 (steady problem)
  // in order to not stop pseudo-timestepping approach before having converged
  double const start_time = 0.0;
  double const end_time   = (test_case == 1) ? 1000.0 : 8.0;

  unsigned int refine_level = 0;

  // superimpose random perturbations at inflow
  bool const use_perturbation = false;
  // amplitude of perturbations relative to maximum velocity on centerline
  double const amplitude_perturbation = 0.25;

  unsigned int const n_points_y = 10;
  unsigned int const n_points_z = dim == 3 ? n_points_y : 1;

  std::vector<double> y_values = std::vector<double>(n_points_y);
  std::vector<double> z_values = std::vector<double>(n_points_z);

  std::vector<dealii::Tensor<1, dim, double>> velocity_values =
    std::vector<dealii::Tensor<1, dim, double>>(n_points_y * n_points_z);

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-6;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = problem_type;
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
    this->param.time_integrator_oif             = TimeIntegratorOIF::ExplRK2Stage2;
    this->param.order_time_integrator           = 3;
    this->param.start_with_low_order            = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = Um;
    this->param.cfl                             = cfl_number;
    this->param.cfl_oif                         = this->param.cfl;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-3;
    this->param.time_step_size_max              = 1.0e-2;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 8.0;

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
    this->param.divergence_penalty_factor                  = 1.0e0;
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
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::CG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
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
    this->refine_level = this->param.grid.n_refine_global;

    auto const lambda_create_coarse_triangulation = [&](dealii::Triangulation<dim, dim> & tria) {
      create_coarse_grid<dim>(tria, this->grid->periodic_faces, cylinder_type_string);
    };

    this->grid->create_triangulation(this->param.grid, lambda_create_coarse_triangulation);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0,
           new InflowBC<dim>(
             Um, H, end_time, test_case, use_perturbation, y_values, z_values, velocity_values)));
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->neumann_bc.insert(2);
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

    std::string name = this->output_name + "_l" + std::to_string(this->refine_level) + "_k" +
                       std::to_string(this->param.degree_u);

    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = name;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_surface_mesh = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_grid         = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // lift and drag
    pp_data.lift_and_drag_data.calculate = true;
    pp_data.lift_and_drag_data.viscosity = viscosity;

    double const U = Um * (dim == 2 ? 2. / 3. : 4. / 9.);
    if(dim == 2)
      pp_data.lift_and_drag_data.reference_value = 1.0 / 2.0 * pow(U, 2.0) * D;
    else if(dim == 3)
      pp_data.lift_and_drag_data.reference_value = 1.0 / 2.0 * pow(U, 2.0) * D * H;

    // surface for calculation of lift and drag coefficients has boundary_ID = 2
    pp_data.lift_and_drag_data.boundary_IDs.insert(2);

    pp_data.lift_and_drag_data.directory     = this->output_directory;
    pp_data.lift_and_drag_data.filename_lift = name + "_lift";
    pp_data.lift_and_drag_data.filename_drag = name + "_drag";

    // pressure difference
    pp_data.pressure_difference_data.calculate = true;
    if(dim == 2)
    {
      dealii::Point<dim> point_1_2D((X_C - D / 2.0), Y_C), point_2_2D((X_C + D / 2.0), Y_C);
      pp_data.pressure_difference_data.point_1 = point_1_2D;
      pp_data.pressure_difference_data.point_2 = point_2_2D;
    }
    else if(dim == 3)
    {
      dealii::Point<dim> point_1_3D((X_C - D / 2.0), Y_C, H / 2.0),
        point_2_3D((X_C + D / 2.0), Y_C, H / 2.0);
      pp_data.pressure_difference_data.point_1 = point_1_3D;
      pp_data.pressure_difference_data.point_2 = point_2_3D;
    }

    pp_data.pressure_difference_data.directory = this->output_directory;
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
