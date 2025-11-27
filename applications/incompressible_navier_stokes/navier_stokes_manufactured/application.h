/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_

// ExaDG
#include <exadg/grid/mesh_movement_functions.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Manufactured solution for incompressible flow of a generalized Newtonian fluid in a hypercube.
 * (Navier-)Stokes equations, where the convective term may be disabled
 * u     ... velocity vector
 * p     ... kinematic pressure
 * nu    ... kinematic viscosity
 * rho   ... density
 * f     ... body force vector
 * sigma ... Cauchy stress
 *
 * d/dt(u) + (grad(u)) * u - 1/rho * div(sigma) = f
 *
 * with
 *
 * sigma := -p * I + 2 * nu * sym_grad(u)
 *
 * such that with variable viscosity we have
 *
 * d/dt(u) + (grad(u)) * u + grad(p) - nu * div(grad(u)) - 2 * sym_grad(u) * grad(nu) = f
 *
 * In 2D, we derive a solution by setting
 *
 * u1 = sin(x) * cos(y) * cos(t)
 *
 * p  = cos(x*y) * cos(t)
 *
 * and derive
 *
 * u2 = -cos(t) * cos(x) * sin(y)
 *
 * shear_rate := sqrt(2 * sym_grad(u) : sym_grad(u)) = 2 cos(t) * cos(x) * cos(y)
 *
 * and insert into the generalized Carreau-Yasuda law
 *
 * nu := nu_oo + (nu_0 - nu_oo) * [kappa + (lambda*shear_rate)^a]^[(n-1)/a]
 *
 * to derive a suitable body force vector f from the momentum balance equation.
 */

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double const t     = this->get_time();
    double const x     = p[0];
    double const y     = p[1];
    double const sin_x = std::sin(x);
    double const sin_y = std::sin(y);
    double const cos_x = std::cos(x);
    double const cos_y = std::cos(y);
    double const cos_t = std::cos(t);

    if(component == 0)
      return sin_x * cos_y * cos_t;
    else if(component == 1)
      return -cos_t * cos_x * sin_y;
    else
    {
      AssertThrow(component < 2, dealii::ExcMessage("Manufactured solution only valid for 2D."));
      return 0.0;
    }
  }
};


template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure() : dealii::Function<dim>(1, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double const t      = this->get_time();
    double const x      = p[0];
    double const y      = p[1];
    double const cos_xy = std::cos(x * y);
    double const cos_t  = std::cos(t);

    return cos_xy * cos_t;
  }
};

template<int dim>
class NeumannBoundaryVelocity : public dealii::Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const &        formulation_viscous_term,
                          double const &                        normal_x,
                          double const &                        normal_y,
                          double const &                        nu_oo,
                          GeneralizedNewtonianModelData const & data)
    : dealii::Function<dim>(dim, 0.0),
      formulation_viscous_term(formulation_viscous_term),
      normal_x(normal_x),
      normal_y(normal_y),
      nu_oo(nu_oo),
      data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    /*
     * This is the kinematic traction vector t obtained via integration by
     * parts of the stress term yielding for the stress-divergence formulation
     *
     * t = (-p * I + 2 * nu * sym_grad(u)) * n
     *
     * or using pseudo-tractions in the Laplacian formulation
     *
     * t* := (-p * I + nu * grad(u)) * n
     *
     * where the pressure contribution is considered in the pressure operator.
     */

    double const t     = this->get_time();
    double const x     = p[0];
    double const y     = p[1];
    double const sin_x = std::sin(x);
    double const sin_y = std::sin(y);
    double const cos_x = std::cos(x);
    double const cos_y = std::cos(y);
    double const cos_t = std::cos(t);

    double const du1_dx = cos_x * cos_y * cos_t;
    double const du1_dy = -sin_x * sin_y * cos_t;
    double const du2_dx = cos_t * sin_x * sin_y;
    double const du2_dy = -cos_t * cos_x * cos_y;

    // rheological law to obtain the kinematic viscosity nu
    // nu = nu_oo + (nu_0 - nu_oo) * [kappa + (lambda*shear_rate)^a]^[(n-1)/a]
    double const shear_rate = 2.0 * cos_t * cos_x * cos_y;
    double const kinematic_viscosity =
      nu_oo +
      data.viscosity_margin *
        std::pow(data.kappa + std::pow(data.lambda * shear_rate, data.a), (data.n - 1.0) / data.a);

    dealii::Tensor<2, dim> grad_u;
    grad_u[0][0] = du1_dx;
    grad_u[0][1] = du1_dy;
    grad_u[1][0] = du2_dx;
    grad_u[1][1] = du2_dy;

    dealii::Tensor<1, dim> normal;
    normal[0] = normal_x;
    normal[1] = normal_y;

    dealii::Tensor<1, dim> traction = grad_u * normal;

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      traction += transpose(grad_u) * normal;
    }

    traction *= kinematic_viscosity;

    AssertThrow(component < 2, dealii::ExcMessage("Manufactured solution only valid for 2D."));

    return traction[component];
  }

private:
  FormulationViscousTerm const  formulation_viscous_term;
  double const                  normal_x;
  double const                  normal_y;
  double const                  nu_oo;
  GeneralizedNewtonianModelData data;
};

template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(bool const                            include_convective_term,
                double const &                        nu_oo,
                GeneralizedNewtonianModelData const & data)
    : dealii::Function<dim>(dim, 0.0),
      include_convective_term(include_convective_term),
      nu_oo(nu_oo),
      data(data)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    /*
     * The solution laid out above is inserted into the momentum balance
     * equation to derive the vector f on the right-hand side.
     *
     * f = d/dt(u) + (grad(u)) * u + grad(p)
     *     - nu * div(grad(u)) - 2 * sym_grad(u) * grad(nu)
     */

    double const t      = this->get_time();
    double const x      = p[0];
    double const y      = p[1];
    double const sin_x  = std::sin(x);
    double const sin_y  = std::sin(y);
    double const cos_x  = std::cos(x);
    double const cos_y  = std::cos(y);
    double const sin_t  = std::sin(t);
    double const cos_t  = std::cos(t);
    double const sin_xy = std::sin(x * y);

    double const u1 = sin_x * cos_y * cos_t;
    double const u2 = -cos_t * cos_x * sin_y;

    double const du1_dt = -sin_x * cos_y * sin_t;
    double const du1_dx = cos_x * cos_y * cos_t;
    double const du1_dy = -sin_x * sin_y * cos_t;
    double const du2_dt = sin_t * cos_x * sin_y;
    double const du2_dx = cos_t * sin_x * sin_y;
    double const du2_dy = -cos_t * cos_x * cos_y;

    double const du1_dxx = -sin_x * cos_y * cos_t;
    double const du1_dyy = -sin_x * cos_y * cos_t;
    double const du2_dxx = cos_t * cos_x * sin_y;
    double const du2_dyy = cos_t * cos_x * sin_y;

    double const dp_dx = -sin_xy * cos_t * y;
    double const dp_dy = -sin_xy * cos_t * x;

    dealii::Tensor<2, dim> grad_u;
    grad_u[0][0] = du1_dx;
    grad_u[0][1] = du1_dy;
    grad_u[1][0] = du2_dx;
    grad_u[1][1] = du2_dy;

    dealii::Tensor<1, dim> div_grad_u;
    div_grad_u[0] = du1_dxx + du1_dyy;
    div_grad_u[1] = du2_dxx + du2_dyy;

    // add time derivative terms
    dealii::Tensor<1, dim> rhs;
    rhs[0] += du1_dt;
    rhs[1] += du2_dt;

    // convective term
    if(include_convective_term)
    {
      dealii::Tensor<1, dim> u;
      u[0] = u1;
      u[1] = u2;
      rhs += grad_u * u;
    }

    // pressure gradient
    rhs[0] += dp_dx;
    rhs[1] += dp_dy;

    // velocity Laplacian respecting the rheological law to obtain the kinematic viscosity
    double const shear_rate = 2.0 * cos_t * cos_x * cos_y;
    double const kinematic_viscosity =
      nu_oo +
      data.viscosity_margin *
        std::pow(data.kappa + std::pow(data.lambda * shear_rate, data.a), (data.n - 1.0) / data.a);
    rhs -= kinematic_viscosity * div_grad_u;

    // viscous term from variable viscosity
    dealii::Tensor<1, dim> grad_nu;
    grad_nu[0] = -2.0 * cos_t * sin_x * cos_y; // d_shear_rate_dx
    grad_nu[1] = -2.0 * cos_t * cos_x * sin_y; // d_shear_rate_dy
    grad_nu *= data.lambda * (data.n - 1) * std::pow(data.lambda * shear_rate, data.a - 1.0) *
               data.viscosity_margin *
               std::pow(data.kappa + std::pow(data.lambda * shear_rate, data.a),
                        (data.n - 1.0 - data.a) / data.a);

    rhs -= (grad_u + transpose(grad_u)) * grad_nu;

    AssertThrow(component < 2, dealii::ExcMessage("Manufactured solution only valid for 2D."));

    return rhs[component];
  }

private:
  bool const                    include_convective_term;
  double const                  nu_oo;
  GeneralizedNewtonianModelData data;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      // clang-format off
      prm.add_parameter("MoveGrid",                            move_grid,                                         "Should the grid be deformed over time?");
      prm.add_parameter("WriteRestart",                        write_restart,                                     "Should restart files be written?");
      prm.add_parameter("ReadRestart",                         read_restart,                                      "Is this a restarted simulation?");
      prm.add_parameter("IncludeConvectiveTerm",               include_convective_term,                           "Include the nonlinear convective term.",          dealii::Patterns::Bool());
      prm.add_parameter("PureDirichletProblem",                pure_dirichlet_problem,                            "Solve a pure Dirichlet problem.",                 dealii::Patterns::Bool());
      prm.add_parameter("StartTime",                           start_time,                                        "Simulation start time.",                          dealii::Patterns::Double());
      prm.add_parameter("EndTime",                             end_time,                                          "Simulation end time.",                            dealii::Patterns::Double());
      prm.add_parameter("IntervalStart",                       interval_start,                                    "Hypercube domain start.",                         dealii::Patterns::Double());
      prm.add_parameter("IntervalEnd",                         interval_end,                                      "Hypercube domain end.",                           dealii::Patterns::Double());
      prm.add_parameter("Density",                             density,                                           "Incompressible model: density.",                  dealii::Patterns::Double());
      prm.add_parameter("KinematicViscosity",                  kinematic_viscosity,                               "Newtonian model: kinematic_viscosity.",           dealii::Patterns::Double());
      prm.add_parameter("UseGeneralizedNewtonianModel",        generalized_newtonian_model_data.is_active,        "Use generalized Newtonian or Newtonian model.",   dealii::Patterns::Bool());
      prm.add_parameter("FormulationViscousTerm",              formulation_viscous_term,                          "Formulation of the viscous term.");
      prm.add_parameter("TemporalDiscretization",              temporal_discretization,                           "Temporal discretization.");
      prm.add_parameter("TreatmentOfConvectiveTerm",           treatment_of_convective_term,                      "Treatment of the convective term.");
      prm.add_parameter("TreatmentOfVariableViscosity",        treatment_of_variable_viscosity,                   "Treatment of the variable viscosity.");
      prm.add_parameter("GeneralizedNewtonianViscosityMargin", generalized_newtonian_model_data.viscosity_margin, "Generalized Newtonian models: viscosity margin.", dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianKappa",           generalized_newtonian_model_data.kappa,            "Generalized Newtonian models: kappa.",            dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianLambda",          generalized_newtonian_model_data.lambda,           "Generalized Newtonian models: lambda.",           dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianA",               generalized_newtonian_model_data.a,                "Generalized Newtonian models: a.",                dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianN",               generalized_newtonian_model_data.n,                "Generalized Newtonian models: n.",                dealii::Patterns::Double());
      // clang-format off
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = ProblemType::Unsteady;
    this->param.equation_type =
      include_convective_term ? EquationType::NavierStokes : EquationType::Stokes;

    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    this->param.formulation_viscous_term = formulation_viscous_term;
    this->param.right_hand_side          = true;

    // ALE
    this->param.ale_formulation                     = move_grid;
    this->param.mesh_movement_type                  = MeshMovementType::Function;
    this->param.neumann_with_variable_normal_vector = true;

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = kinematic_viscosity;
    this->param.density    = density;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = temporal_discretization;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = std::abs(end_time - start_time) / 100;
    this->param.order_time_integrator         = 2;     // 1; // 2; // 3;
    this->param.start_with_low_order          = false; // true;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;


    // SPATIAL DISCRETIZATION
    this->param.spatial_discretization      = SpatialDiscretization::L2;
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    this->param.treatment_of_convective_term = treatment_of_convective_term;
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

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
    if(pure_dirichlet_problem)
      this->param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalMeanValue;

    // div-div and continuity penalty terms
    this->param.use_divergence_penalty                     = true;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // TURBULENCE
    this->param.turbulence_model_data.is_active        = use_turbulence_model;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;

    this->param.treatment_of_variable_viscosity = treatment_of_variable_viscosity;

    // GENERALIZED NEWTONIAN MODEL
    generalized_newtonian_model_data.generalized_newtonian_model =
      GeneralizedNewtonianViscosityModel::GeneralizedCarreauYasuda;
    this->param.generalized_newtonian_model_data = generalized_newtonian_model_data;

    if(use_turbulence_model || generalized_newtonian_model_data.is_active)
    {
      this->param.use_cell_based_face_loops = false;
      this->param.quad_rule_linearization   = QuadratureRuleLinearization::Standard;
    }

    // RESTART
    this->param.restarted_simulation       = read_restart;
    this->param.restart_data.write_restart = write_restart;
    // write restart every 40% of the simulation time
    this->param.restart_data.n_snapshots_keep = 10;
    this->param.restart_data.interval_time = (this->param.end_time - this->param.start_time) * 0.01;
    this->param.restart_data.interval_time_start =
      (this->param.end_time - this->param.start_time) * 0.8;
    this->param.restart_data.interval_time_end =
      (this->param.end_time - this->param.start_time) * 0.9;
    this->param.restart_data.directory_coarse_triangulation = this->output_parameters.directory;
    this->param.restart_data.directory                      = this->output_parameters.directory;
    this->param.restart_data.filename            = this->output_parameters.filename + "_restart";
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;

    this->param.restart_data.discretization_identical                        = false;
    this->param.restart_data.consider_mapping_write                          = true;
    this->param.restart_data.consider_mapping_read_source                    = true;
    this->param.restart_data.consider_restart_time_in_mesh_movement_function = true;

    this->param.restart_data.rpe_rtree_level            = 0;
    this->param.restart_data.rpe_tolerance_unit_cell    = 1e-2;
    this->param.restart_data.rpe_enforce_unique_mapping = false;

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

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplitting)
    {
      this->param.solver_momentum =
        treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ?
          SolverMomentum::CG :
          SolverMomentum::FGMRES;
      this->param.solver_data_momentum = SolverData(1000, 1e-12, 1e-8);
      this->param.preconditioner_momentum = MomentumPreconditioner::Multigrid;
    }


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = this->param.order_time_integrator - 1;
    this->param.rotational_formulation       = true;

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // Newton solver
      this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

      // linear solver
      this->param.solver_momentum                = SolverMomentum::GMRES;
      this->param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
      this->param.preconditioner_momentum        = MomentumPreconditioner::Multigrid;
      this->param.update_preconditioner_momentum = true;
    }

    if(this->param.preconditioner_momentum == MomentumPreconditioner::Multigrid)
    {
      // clang-format off
      this->param.multigrid_data_momentum.type       = MultigridType::cphMG;
      this->param.multigrid_data_momentum.p_sequence = PSequenceType::DecreaseByOne;

      this->param.multigrid_operator_type_momentum                        = MultigridOperatorType::ReactionDiffusion;
      this->param.multigrid_data_momentum.smoother_data.iterations        = 5;
      this->param.multigrid_data_momentum.smoother_data.smoother          = MultigridSmoother::Chebyshev; // MultigridSmoother::Jacobi
      this->param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.8; // Jacobi,    default: 0.8
      this->param.multigrid_data_momentum.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
      this->param.multigrid_data_momentum.smoother_data.iterations_eigenvalue_estimation = 60; // Chebyshev, default: 20

      this->param.multigrid_data_momentum.coarse_problem.solver         = this->param.non_explicit_convective_problem() ? MultigridCoarseGridSolver::GMRES : MultigridCoarseGridSolver::CG; // MultigridCoarseGridSolver::AMG
      this->param.multigrid_data_momentum.coarse_problem.solver_data    = SolverData(1000, 1e-12, 1e-8, 30);
      this->param.multigrid_data_momentum.smoother_data.preconditioner  = PreconditionerSmoother::PointJacobi;
      this->param.multigrid_data_momentum.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
      this->param.multigrid_data_momentum.coarse_problem.amg_data.ml_data.smoother_sweeps = 2;
      this->param.multigrid_data_momentum.coarse_problem.amg_data.ml_data.n_cycles        = 2;
      this->param.multigrid_data_momentum.coarse_problem.amg_data.ml_data.smoother_type   = "Chebyshev"; // "ILU"
      this->param.multigrid_data_momentum.coarse_problem.amg_data.ml_data.coarse_type     = "Amesos-KLU";
#endif
      // clang-format on
    }

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;

    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit &&
       include_convective_term == true)
      this->param.multigrid_operator_type_velocity_block =
        MultigridOperatorType::ReactionConvectionDiffusion;
    else
      this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;

    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi; ////BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
      this->param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
    else
      this->param.preconditioner_pressure_block =
        SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;

        dealii::GridGenerator::hyper_cube(tria, interval_start, interval_end, true);

        // Save the *coarse* triangulation for later deserialization.
        if(write_restart and this->param.grid.triangulation_type == TriangulationType::Serial)
        {
          save_coarse_triangulation<dim>(this->param.restart_data, tria);
        }

        if(vector_local_refinements.size() > 0)
          refine_local(tria, vector_local_refinements);

        if(global_refinements > 0)
          tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion;

    if(move_grid)
    {
      MeshMovementData<dim> data;
      data.temporal                       = MeshMovementAdvanceInTime::Sin;
      data.shape                          = MeshMovementShape::Sin;
      data.dimensions[0]                  = std::abs(interval_end - interval_start);
      data.dimensions[1]                  = std::abs(interval_end - interval_start);
      data.amplitude                      = std::abs(interval_end - interval_start) / 15.0;
      data.period                         = std::abs(end_time - start_time);
      data.t_start                        = start_time;
      data.t_end                          = end_time;
      data.spatial_number_of_oscillations = 1.0;
      mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));
    }
    return mesh_motion;
  }

  void
  set_boundary_descriptor() final
  {
    AssertThrow(dim == 2,
                dealii::ExcMessage("Manufactured solution for dim == 2 implemented only."));

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    double const normal_x = -1.0;
    double const normal_y = 0.0;

    for(unsigned int i = 0; i < 2 * dim; ++i)
    {
      if(i == 0 && pure_dirichlet_problem == false)
      {
        // Neumann boundary condition for boundary with id 0.
        this->boundary_descriptor->velocity->neumann_bc.insert(
          pair(i,
               new NeumannBoundaryVelocity<dim>(this->param.formulation_viscous_term,
                                                normal_x,
                                                normal_y,
                                                kinematic_viscosity,
                                                generalized_newtonian_model_data)));

        this->boundary_descriptor->pressure->dirichlet_bc.insert(
          pair(i, new AnalyticalSolutionPressure<dim>()));
      }
      else
      {
        // Dirichlet boundary conditions.
        this->boundary_descriptor->velocity->dirichlet_bc.insert(
          pair(i, new AnalyticalSolutionVelocity<dim>()));
        this->boundary_descriptor->pressure->neumann_bc.insert(i);
      }
    }
  }

  void
  set_field_functions() final
  {
    AssertThrow(dim == 2,
                dealii::ExcMessage("Manufactured solution for dim == 2 implemented only."));

    this->field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    this->field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>());
    this->field_functions->analytical_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>());
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(
      include_convective_term, kinematic_viscosity, generalized_newtonian_model_data));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 100.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_shear_rate   = true;
    pp_data.output_data.write_viscosity    = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = true;

    // calculation of velocity error
    pp_data.error_data_u.write_errors_to_file               = true;
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = true; // false;
    pp_data.error_data_u.name                      = "velocity";
    pp_data.error_data_u.directory                 = this->output_parameters.directory;

    // ... pressure error
    pp_data.error_data_p.write_errors_to_file               = true;
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = true; // false;
    pp_data.error_data_p.name                      = "pressure";
    pp_data.error_data_p.directory                 = this->output_parameters.directory;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  bool read_restart  = false;
  bool write_restart = false;
  bool move_grid     = false;

  bool   include_convective_term = true;
  bool   pure_dirichlet_problem  = true;
  double start_time              = 0.0;
  double end_time                = 0.1;
  double density                 = 1000.0;
  double kinematic_viscosity     = 5e-6;
  bool   use_turbulence_model    = false;

  FormulationViscousTerm formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  TemporalDiscretization temporal_discretization  = TemporalDiscretization::Undefined;
  TreatmentOfConvectiveTerm treatment_of_convective_term =
    TreatmentOfConvectiveTerm::LinearlyImplicit;
  TreatmentOfVariableViscosity treatment_of_variable_viscosity =
    TreatmentOfVariableViscosity::Explicit;

  GeneralizedNewtonianModelData generalized_newtonian_model_data;
  double                        interval_start = 0.0;
  double                        interval_end   = 0.1;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_ */
