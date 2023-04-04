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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/user_interface/enum_types.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Manufactured solution for incompressible flow of a generalized Newtonian fluid in a hypercube.
 * (Navier-)Stokes equations:
 * c     ... scaling the convective term with c = 0 or c = 1
 * u     ... velocity vector
 * p     ... kinematic pressure
 * nu    ... kinematic viscosity
 * rho   ... density
 * f     ... body force vector
 * sigma ... Cauchy stress
 *
 * d/dt(u) + c (grad(u)) * u - 1/rho * div(sigma) = f
 *
 * with
 *
 * sigma := -p * I + 2 * nu * sym_grad(u)
 *
 * such that with variable viscosity we have
 *
 * d/dt(u) + c (grad(u)) * u + grad(p) - nu * div(grad(u)) - 2 * sym_grad(u) * grad(nu) = f
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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double t     = this->get_time();
    double x     = p[0];
    double y     = p[1];
    double sin_x = std::sin(x);
    double sin_y = std::sin(y);
    double cos_x = std::cos(x);
    double cos_y = std::cos(y);
    double cos_t = std::cos(t);

    double result = 0.0;

    if(component == 0)
      result = sin_x * cos_y * cos_t;
    else if(component == 1)
      result = -cos_t * cos_x * sin_y;

    return result;
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
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double t      = this->get_time();
    double x      = p[0];
    double y      = p[1];
    double cos_xy = std::cos(x * y);
    double cos_t  = std::cos(t);

    return cos_xy * cos_t;
  }
};

template<int dim>
class NeumannBoundaryVelocity : public dealii::Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const & formulation_viscous_term,
                          double const &                 normal_x,
                          double const &                 normal_y,
                          double const &                 nu_oo,
                          double const &                 nu_margin,
                          double const &                 kappa,
                          double const &                 lambda,
                          double const &                 a,
                          double const &                 n)
    : dealii::Function<dim>(dim, 0.0),
      formulation_viscous_term(formulation_viscous_term),
      normal_x(normal_x),
      normal_y(normal_y),
      nu_oo(nu_oo),
      nu_0(nu_margin - nu_oo),
      kappa(kappa),
      lambda(lambda),
      a(a),
      n(n)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
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
     */

    double t      = this->get_time();
    double x      = p[0];
    double y      = p[1];
    double sin_x  = std::sin(x);
    double sin_y  = std::sin(y);
    double cos_x  = std::cos(x);
    double cos_y  = std::cos(y);
    double cos_t  = std::cos(t);
    double cos_xy = std::cos(x * y);

    double du1_dx = cos_x * cos_y * cos_t;
    double du1_dy = -sin_x * sin_y * cos_t;
    double du2_dx = cos_t * sin_x * sin_y;
    double du2_dy = -cos_t * cos_x * cos_y;

    double pressure = cos_xy * cos_t;

    // rheological law to obtain the kinematic viscosity nu
    // nu = nu_oo + (nu_0 - nu_oo) * [kappa + (lambda*shear_rate)^a]^[(n-1)/a]
    double shear_rate = 2.0 * cos_t * cos_x * cos_y;
    double kinematic_viscosity =
      nu_oo + (nu_0 - nu_oo) * std::pow(kappa + std::pow(lambda * shear_rate, a), (n - 1.0) / a);

    dealii::Tensor<2, dim> grad_u;
    grad_u[0][0] = du1_dx;
    grad_u[0][1] = du1_dy;
    grad_u[1][0] = du2_dx;
    grad_u[1][1] = du2_dy;

    dealii::Tensor<1, dim> normal;
    normal[0] = normal_x;
    normal[1] = normal_y;

    dealii::Tensor<1, dim> traction;

    traction += grad_u * normal;

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      traction += transpose(grad_u) * normal;
    }

    traction *= kinematic_viscosity;

    traction[0] = -pressure;
    traction[1] = -pressure;

    return traction[component];
  }

private:
  FormulationViscousTerm const formulation_viscous_term;
  double const                 normal_x;
  double const                 normal_y;
  double const                 nu_oo;
  double const                 nu_0;
  double const                 kappa;
  double const                 lambda;
  double const                 a;
  double const                 n;
};

template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(bool const     include_convective_term,
                double const & normal_x,
                double const & normal_y,
                double const & nu_oo,
                double const & nu_margin,
                double const & kappa,
                double const & lambda,
                double const & a,
                double const & n)
    : dealii::Function<dim>(dim, 0.0),
      include_convective_term(include_convective_term),
      normal_x(normal_x),
      normal_y(normal_y),
      nu_oo(nu_oo),
      nu_0(nu_margin - nu_oo),
      kappa(kappa),
      lambda(lambda),
      a(a),
      n(n)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    /*
     * The solution laid out above is inserte into the momentum balance
     * equation to derive the vector f on the right-hand side.
     *
     * f = d/dt(u) + c (grad(u)) * u + grad(p)
     *     - nu * div(grad(u)) - 2 * sym_grad(u) * grad(nu)
     */

    double t      = this->get_time();
    double x      = p[0];
    double y      = p[1];
    double sin_x  = std::sin(x);
    double sin_y  = std::sin(y);
    double cos_x  = std::cos(x);
    double cos_y  = std::cos(y);
    double sin_t  = std::sin(t);
    double cos_t  = std::cos(t);
    double sin_xy = std::sin(x * y);

    double u1 = sin_x * cos_y * cos_t;
    double u2 = -cos_t * cos_x * sin_y;

    double du1_dt = -sin_x * cos_y * sin_t;
    double du1_dx = cos_x * cos_y * cos_t;
    double du1_dy = -sin_x * sin_y * cos_t;
    double du2_dt = sin_t * cos_x * sin_y;
    double du2_dx = cos_t * sin_x * sin_y;
    double du2_dy = -cos_t * cos_x * cos_y;

    double du1_dxx = -sin_x * cos_y * cos_t;
    double du1_dyy = -sin_x * cos_y * cos_t;
    double du2_dxx = cos_t * cos_x * sin_y;
    double du2_dyy = cos_t * cos_x * sin_y;

    double dp_dx = -sin_xy * cos_t * y;
    double dp_dy = -sin_xy * cos_t * x;

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

    // velocity Laplacian respecting the
    // rheological law to obtain the kinematic viscosity
    // nu_oo + (nu_0 - nu_oo) * [kappa + (lambda*shear_rate)^a]^[(n-1)/a]
    double shear_rate = 2.0 * cos_t * cos_x * cos_y;
    double kinematic_viscosity =
      nu_oo + (nu_0 - nu_oo) * std::pow(kappa + std::pow(lambda * shear_rate, a), (n - 1.0) / a);
    rhs -= kinematic_viscosity * div_grad_u;

    // viscous term from variable viscosity
    dealii::Tensor<1, dim> grad_nu;
    grad_nu[0] = -2.0 * cos_t * sin_x * cos_y; // d_shear_rate_dx
    grad_nu[1] = -2.0 * cos_t * cos_x * sin_y; // d_shear_rate_dy
    grad_nu *= lambda * (n - 1) * std::pow(lambda * shear_rate, a - 1.0) * (nu_0 - nu_oo) *
               std::pow(kappa + std::pow(lambda * shear_rate, a), (n - 2.0) / a);

    rhs += (grad_u + transpose(grad_u)) * grad_nu;

    return rhs[component];
  }

private:
  bool const   include_convective_term;
  double const normal_x;
  double const normal_y;
  double const nu_oo;
  double const nu_0;
  double const kappa;
  double const lambda;
  double const a;
  double const n;
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

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("IncludeConvectiveTerm",
                        include_convective_term,
                        "Include the nonlinear convective term.",
                        dealii::Patterns::Bool());
      prm.add_parameter("PureDirichletProblem",
					    pure_dirichlet_problem,
					    "Solve a pure Dirichlet problem.",
					    dealii::Patterns::Bool());
      prm.add_parameter("StartTime",
    		            start_time,
					    "Simulation start time.",
					    dealii::Patterns::Double());
      prm.add_parameter("EndTime",
    		            end_time,
					    "Simulation end time.",
					    dealii::Patterns::Double());
      prm.add_parameter("IntervalStart",
    		            interval_start,
					    "Hypercube domain start.",
					    dealii::Patterns::Double());
      prm.add_parameter("IntervalEnd",
    		            interval_end,
					    "Hypercube domain end.",
					    dealii::Patterns::Double());
      prm.add_parameter("Density",
          		        density,
      					"Incompressible model: density.",
      					dealii::Patterns::Double());
      prm.add_parameter("KinematicViscosity",
          		        kinematic_viscosity,
      					"Newtonian model: kinematic_viscosity.",
      					dealii::Patterns::Double());
	  prm.add_parameter("UseGeneralizedNewtonianModel",
			            use_generalized_newtonian_model,
						"Use generalized Newtonian model, else Newtonian one.",
						dealii::Patterns::Bool());
	  prm.add_parameter("FormulationViscousTermLaplaceFormulation",
			            formulation_viscous_term_laplace_formulation,
						"Use Laplace or Divergence formulation for viscous term.",
						dealii::Patterns::Bool());
	  prm.add_parameter("TemporalDiscretization",
			            temporal_discretization_string,
						"Temporal discretization.",
						dealii::Patterns::Selection("BDFCoupledSolution|BDFPressureCorrection|BDFDualSplittingScheme"));
	  prm.add_parameter("TreatmentOfConvectiveTermImplicit",
			            treatment_of_convective_term_implicit,
						"Treat convective term implicit, else explicit",
						dealii::Patterns::Bool());
	  prm.add_parameter("TreatmentOfVariableViscosityImplicit",
			            treatment_of_variable_viscosity_implicit,
						"Treat the variable viscosity implicit or extrapolate in time.",
						dealii::Patterns::Bool());
      prm.add_parameter("GeneralizedNewtonianViscosityMargin",
    		            generalized_newtonian_viscosity_margin,
					    "Generalized Newtonian models: viscosity margin.",
					    dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianKappa",
    		             generalized_newtonian_kappa,
      					 "Generalized Newtonian models: kappa.",
      					 dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianLambda",
    		             generalized_newtonian_lambda,
      					 "Generalized Newtonian models: lambda.",
      					 dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianA",
    		             generalized_newtonian_a,
      					 "Generalized Newtonian models: a.",
      					 dealii::Patterns::Double());
      prm.add_parameter("GeneralizedNewtonianN",
    		             generalized_newtonian_n,
      					 "Generalized Newtonian models: n.",
      					 dealii::Patterns::Double());
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    // clang-format off
    if     (temporal_discretization_string == "BDFCoupledSolution")     temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
    else if(temporal_discretization_string == "BDFPressureCorrection")  temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
    else if(temporal_discretization_string == "BDFDualSplittingScheme") temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
    else AssertThrow(false, dealii::ExcMessage("Unknown temporal discretization. Not implemented."));
    // clang-format on
  }
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = ProblemType::Unsteady;
    if(include_convective_term == true)
      this->param.equation_type = EquationType::NavierStokes;
    else
      this->param.equation_type = EquationType::Stokes;

    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    FormulationViscousTerm formulation_viscous_term =
      formulation_viscous_term_laplace_formulation ? FormulationViscousTerm::LaplaceFormulation :
                                                     FormulationViscousTerm::DivergenceFormulation;
    this->param.formulation_viscous_term = formulation_viscous_term;
    this->param.right_hand_side          = true;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = kinematic_viscosity;
    this->param.density    = density;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = temporal_discretization;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = this->param.end_time;
    this->param.order_time_integrator         = 2;     // 1; // 2; // 3;
    this->param.start_with_low_order          = false; // true;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    this->param.treatment_of_convective_term = treatment_of_convective_term_implicit ?
                                                 TreatmentOfConvectiveTerm::Implicit :
                                                 TreatmentOfConvectiveTerm::Explicit;
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

    this->param.treatment_of_variable_viscosity =
      treatment_of_variable_viscosity_implicit ?
        TreatmentOfVariableViscosity::Implicit :
        TreatmentOfVariableViscosity::LinearizedInTimeImplicit;

    // GENERALIZED NEWTONIAN MODEL
    this->param.generalized_newtonian_model_data.is_active = use_generalized_newtonian_model;
    if(use_generalized_newtonian_model)
    {
      this->param.generalized_newtonian_model_data.generalized_newtonian_model =
        generalized_newtonian_model;
      this->param.generalized_newtonian_model_data.viscosity_margin =
        generalized_newtonian_viscosity_margin;
      this->param.generalized_newtonian_model_data.kappa  = generalized_newtonian_kappa;
      this->param.generalized_newtonian_model_data.lambda = generalized_newtonian_lambda;
      this->param.generalized_newtonian_model_data.a      = generalized_newtonian_a;
      this->param.generalized_newtonian_model_data.n      = generalized_newtonian_n;
    }

    if(use_turbulence_model || use_generalized_newtonian_model)
    {
      this->param.use_cell_based_face_loops = false;
      this->param.quad_rule_linearization   = QuadratureRuleLinearization::Standard;
    }

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

    // formulation
    this->param.order_pressure_extrapolation = this->param.order_time_integrator - 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = true;


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
  create_grid() final
  {
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation,
                                      interval_start,
                                      interval_end,
                                      true);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    AssertThrow(dim == 2,
                dealii::ExcMessage("Manufactured solution for dim == 2 implemented only."));

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    double normal_x = 1.0;
    double normal_y = 0.0;

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
                                                generalized_newtonian_viscosity_margin,
                                                generalized_newtonian_kappa,
                                                generalized_newtonian_lambda,
                                                generalized_newtonian_a,
                                                generalized_newtonian_n)));

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
    double normal_x = 1.0;
    double normal_y = 0.0;

    this->field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    this->field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>());
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->right_hand_side.reset(
      new RightHandSide<dim>(include_convective_term,
                             normal_x,
                             normal_y,
                             kinematic_viscosity,
                             generalized_newtonian_viscosity_margin,
                             generalized_newtonian_kappa,
                             generalized_newtonian_lambda,
                             generalized_newtonian_a,
                             generalized_newtonian_n));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 1.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_shear_rate   = true;
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

    // ... pressure error
    pp_data.error_data_p.write_errors_to_file               = true;
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = true; // false;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  bool                   include_convective_term                      = true;
  bool                   pure_dirichlet_problem                       = true;
  double                 start_time                                   = 0.0;
  double                 end_time                                     = 0.1;
  double                 density                                      = 1000.0;
  double                 kinematic_viscosity                          = 5e-6;
  bool                   use_turbulence_model                         = false;
  bool                   use_generalized_newtonian_model              = true;
  bool                   formulation_viscous_term_laplace_formulation = true;
  TemporalDiscretization temporal_discretization               = TemporalDiscretization::Undefined;
  std::string            temporal_discretization_string = "BDFCoupledSolution";
  bool                   treatment_of_convective_term_implicit = false;
  bool                   treatment_of_variable_viscosity_implicit = false;

  double generalized_newtonian_viscosity_margin = 49.0e-6;
  double generalized_newtonian_kappa            = 1.1;
  double generalized_newtonian_lambda           = 10.0;
  double generalized_newtonian_a                = 2.1;
  double generalized_newtonian_n                = 0.25;
  double interval_start                         = 0.0;
  double interval_end                           = 0.1;

  GeneralizedNewtonianViscosityModel const generalized_newtonian_model =
    GeneralizedNewtonianViscosityModel::GeneralizedCarreauYasuda;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_NAVIER_STOKES_MANUFACTURED_H_ */
