/*
 * stokes_curl_flow.h
 *
 *  Created on: Oct 18, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_CURL_FLOW_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_CURL_FLOW_H_

namespace IncNS
{
namespace StokesCurlFlow
{
template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double result = 0.0;

    double x = p[0];
    double y = p[1];
    if(component == 0)
      result = x * x * (1 - x) * (1 - x) * (2 * y * (1 - y) * (1 - y) - 2 * y * y * (1 - y));
    else if(component == 1)
      result = -1 * y * y * (1 - y) * (1 - y) * (2 * x * (1 - x) * (1 - x) - 2 * x * x * (1 - x));

    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure() : Function<dim>(1 /*n_components*/, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double result = 0.0;

    result = std::pow(p[0], 5.0) + std::pow(p[1], 5.0) - 1.0 / 3.0;

    return result;
  }
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(double const viscosity) : Function<dim>(dim, 0.0), nu(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double x      = p[0];
    double y      = p[1];
    double result = 0.0;

    // clang-format off
     if(component == 0)
     {
       result = -nu * (+ 4 * (1 - x) * (1-x) * y * (1 - y) * (1-y)
                       - 16 * x * (1 - x) * y * (1 - y) * (1-y)
                       + 4 * x * x * y * (1 - y) * (1-y)
                       - 4 * (1 - x) * (1-x) * y * y  * (1 - y)
                       + 16 * x * (1 - x) * y * y  * (1 - y)
                       - 4 * x * x * y * y * (1 - y)
                       - 12 * x * x * (1 - x) * (1 - x) * (1 - y)
                       + 12 * x * x  * (1 - x) * (1-x) * y)
                 + 5 * x * x * x * x;
     }
     else if(component == 1)
     {
       result = -nu * (12 * (1 - x) * y * y * (1 - y) * (1-y)
                       - 12 * x * y * y * (1 - y) * (1-y)
                       - 4 * x * (1 - x) * (1-x) * (1 - y) * (1-y)
                       + 16 * x * (1 - x) * (1-x) * y * (1 - y)
                       - 4 * x * (1 - x) * (1-x) * y * y
                       + 4 * x * x * (1 - x) * (1 - y) * (1-y)
                       - 16 * x * x * (1 - x) * y * (1 - y)
                       + 4 * x * x  * (1 - x) * y * y)
                 + 5 * y * y * y * y;
     }
     else
     {
       AssertThrow(false,
           ExcMessage("Parameter component is invalid for problem with dim=2."));
     }
    // clang-format on

    return result;
  }

private:
  double const nu;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm, true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/stokes_curl_flow/", output_name = "test";

  double const viscosity = 1.0e-5;

  double const start_time = 0.0;
  double const end_time   = 1.0e4;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Steady;
    param.equation_type            = EquationType::Stokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = true;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Steady;
    param.temporal_discretization       = TemporalDiscretization::BDFCoupledSolution;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = 0.1;
    param.order_time_integrator         = 2;    // 1; // 2; // 3;
    param.start_with_low_order          = true; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // pseudo-timestepping
    param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement;
    param.abs_tol_steady = 1e-8;
    param.rel_tol_steady = 1e-6;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
    param.IP_factor_viscous      = 1.0e0;

    // special case: pure DBC's
    param.pure_dirichlet_bc     = true;
    param.adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

    // div-div and continuity penalty
    param.use_divergence_penalty               = true;
    param.divergence_penalty_factor            = 1.0e0;
    param.use_continuity_penalty               = true;
    param.continuity_penalty_factor            = param.divergence_penalty_factor;
    param.continuity_penalty_use_boundary_data = true;
    param.continuity_penalty_components        = ContinuityPenaltyComponents::Normal;
    param.type_penalty_parameter               = TypePenaltyParameter::ViscousAndConvectiveTerms;
    param.apply_penalty_terms_in_postprocessing_step = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_viscous = PreconditionerViscous::Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = param.order_time_integrator - 1;
    param.rotational_formulation       = true;

    // momentum step

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;


    // COUPLED NAVIER-STOKES SOLVER

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid; // InverseMassMatrix;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.type     = MultigridType::phMG;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev; // GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::InverseMassMatrix; // CahouetChabard;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }


  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    const double left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(*triangulation, left, right);
    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    // test case with pure Dirichlet boundary conditions for velocity
    // all boundaries have ID = 0 by default

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>()));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    //  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    //  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->right_hand_side.reset(new RightHandSide<dim>(viscosity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time); // /10;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace StokesCurlFlow
} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_CURL_FLOW_H_ */
