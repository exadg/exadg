/*
 * stokes_shahbazi.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_

namespace ExaDG
{
namespace IncNS
{
namespace StokesShahbazi
{
using namespace dealii;

// perform stability analysis and compute eigenvalue spectrum
// For this analysis one has to use the BDF1 scheme and homogeneous boundary conditions!!!
bool const STABILITY_ANALYSIS = false;

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const nu) : Function<dim>(dim, 0.0), viscosity(nu)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double a      = 2.883356;
    const double lambda = viscosity * (1. + a * a);

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
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(double const nu) : Function<dim>(1, 0.0), viscosity(nu)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double a      = 2.883356;
    const double lambda = viscosity * (1. + a * a);

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

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt(double const nu) : Function<dim>(dim, 0.0), viscosity(nu)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double a      = 2.883356;
    const double lambda = viscosity * (1. + a * a);

    double exp_t  = std::exp(-lambda * t);
    double sin_x  = std::sin(p[0]);
    double cos_x  = std::cos(p[0]);
    double cos_a  = std::cos(a);
    double sin_ay = std::sin(a * p[1]);
    double cos_ay = std::cos(a * p[1]);
    double sinh_y = std::sinh(p[1]);
    double cosh_y = std::cosh(p[1]);
    if(component == 0)
      result = -lambda * exp_t * sin_x * (a * sin_ay - cos_a * sinh_y);
    else if(component == 1)
      result = -lambda * exp_t * cos_x * (cos_ay + cos_a * cosh_y);

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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
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

  std::string output_directory = "output/stokes_shahbazi/", output_name = "test";

  double const viscosity = 1.0e0;

  double const start_time = 0.0;
  double const end_time   = 1.0e-1;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Unsteady;
    param.equation_type            = EquationType::Stokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = param.end_time;
    param.order_time_integrator         = 1; // 1; // 2; // 3;
    if(STABILITY_ANALYSIS)
      param.order_time_integrator = 1;
    param.start_with_low_order = true; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = 1.0; //(param.end_time-param.start_time)/10;


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // gradient term
    param.gradp_integrated_by_parts = true;
    param.gradp_use_boundary_data   = true;
    param.gradp_formulation         = FormulationPressureGradientTerm::Weak;

    // divergence term
    param.divu_integrated_by_parts = true;
    param.divu_use_boundary_data   = true;
    param.divu_formulation         = FormulationVelocityDivergenceTerm::Weak;

    // pressure level is undefined
    param.adjust_pressure_level =
      AdjustPressureLevel::ApplyZeroMeanValue; // ApplyAnalyticalSolutionInPoint;

    // div-div and continuity penalty terms
    param.use_divergence_penalty                     = true;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8);

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
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;


    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = param.order_time_integrator - 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
    param.discretization_of_laplacian   = DiscretizationOfLaplacian::Classical;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    const double left = -1.0, right = 1.0;
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
      pair(0, new AnalyticalSolutionVelocity<dim>(viscosity)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new PressureBC_dudt<dim>(viscosity)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(viscosity));
    field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
    field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = false; // true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time); // /10;
    pp_data.output_data.write_divergence     = false;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(viscosity));
    pp_data.error_data_u.calculate_relative_errors = true; // false;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(viscosity));
    pp_data.error_data_p.calculate_relative_errors = true; // false;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace StokesShahbazi
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_ */
