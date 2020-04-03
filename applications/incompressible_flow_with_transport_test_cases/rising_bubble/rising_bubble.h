/*
 * rising_bubble.h
 *
 *  Created on: Jan 26, 2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_

namespace FTI
{
namespace RisingBubble
{
template<int dim>
class InitialTemperature : public Function<dim>
{
public:
  InitialTemperature(double const T_ref, double const delta_T, double const length)
    : Function<dim>(1, 0.0), T_ref(T_ref), delta_T(delta_T), L(length)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component = 0*/) const
  {
    double radius = 0.0;
    if(dim == 2)
    {
      Point<dim> center = Point<dim>(0.5 * L, 0.35 * L);
      radius            = p.distance(center);
    }
    else if(dim == 3)
    {
      Point<dim> center = Point<dim>(0.5 * L, 0.35 * L, 0.5 * L);
      radius            = p.distance(center);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    double const r_b = 0.25 * L;
    double const factor =
      radius < r_b ? delta_T / 2.0 * (1 + std::cos(numbers::PI * radius / r_b)) : 0.0;

    return (T_ref + delta_T * factor);
  }

private:
  double const T_ref, delta_T, L;
};

template<int dim, typename Number>
class Application : public FTI::ApplicationBase<dim, Number>
{
public:
  Application() : FTI::ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : FTI::ApplicationBase<dim, Number>(input_file)
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

  double const L = 1000.0;

  double const g       = 9.81;
  double const T_ref   = 0.0;
  double const beta    = 1.0 / 300.0;
  double const delta_T = 0.75;

  double const kinematic_viscosity = 1.0e-11;
  double const thermal_diffusivity = kinematic_viscosity;

  double const start_time = 0.0;
  double const end_time   = 1000.0;

  double const CFL_OIF                = 0.4;
  double const CFL                    = CFL_OIF;
  double const max_velocity           = 1.0;
  bool const   adaptive_time_stepping = true;

  // output
  bool const   write_output         = true;
  double const output_interval_time = (end_time - start_time) / 100.0;
  std::string  output_directory = "output/rising_bubble/", output_name = "test";

  void
  set_input_parameters(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = true;
    param.boussinesq_term             = true;


    // PHYSICAL QUANTITIES
    param.start_time                    = start_time;
    param.end_time                      = end_time;
    param.viscosity                     = kinematic_viscosity;
    param.thermal_expansion_coefficient = beta;
    param.reference_temperature         = T_ref;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = max_velocity;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.cfl_oif                         = CFL_OIF;
    param.cfl                             = CFL;
    param.time_step_size                  = 1.0e-1;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Affine;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // special case: pure DBC's
    param.pure_dirichlet_bc = true;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;
    param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = NewtonSolverData(100, 1.e-20, 1.e-6);

    // linear solver
    // use FGMRES for matrix-free BlockJacobi or Multigrid with Krylov methods as smoother/coarse
    // grid solver
    param.solver_momentum         = SolverMomentum::FGMRES;
    param.solver_data_momentum    = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = NewtonSolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void
  set_input_parameters_scalar(ConvDiff::InputParameters & param, unsigned int const scalar_index)
  {
    (void)scalar_index;

    using namespace ConvDiff;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::ConvectionDiffusion;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.analytical_velocity_field   = false;
    param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = thermal_diffusivity;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.adaptive_time_stepping        = adaptive_time_stepping;
    param.order_time_integrator         = 2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.time_step_size                = 1.0e-2;
    param.cfl                           = CFL;
    param.max_velocity                  = max_velocity;
    param.exponent_fe_degree_convection = 1.5;
    param.exponent_fe_degree_diffusion  = 3.0;
    param.diffusion_number              = 0.01;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // mapping
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // SOLVER
    param.solver                    = ConvDiff::Solver::CG;
    param.solver_data               = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner            = Preconditioner::InverseMassMatrix;
    param.multigrid_data.type       = MultigridType::pMG;
    param.multigrid_data.p_sequence = PSequenceType::Bisect;
    param.mg_operator_type          = MultigridOperatorType::ReactionDiffusion;
    param.update_preconditioner     = false;


    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;
    param.filter_solution       = false;
    param.use_overintegration   = true;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    if(dim == 2)
    {
      const double left = 0.0, right = L;
      GridGenerator::hyper_cube(*triangulation, left, right);
    }
    else if(dim == 3)
    {
      std::vector<unsigned int> repetitions({1, 1, 1});
      Point<dim>                point1(0.0, 0.0, 0.0), point2(L, 1.5 * L, L);
      GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
    std::vector<double> gravity = std::vector<double>(dim, 0.0);
    gravity[1]                  = -g;
    field_functions->gravitational_force.reset(new Functions::ConstantFunction<dim>(gravity));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = write_output;
    pp_data.output_data.output_folder        = output_directory + "vtu/";
    pp_data.output_data.output_name          = output_name + "_fluid";
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.degree               = degree;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  void set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<0, dim>> boundary_descriptor,
    unsigned int                                          scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ConstantFunction<dim>(T_ref)));
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new InitialTemperature<dim>(T_ref, delta_T, L));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output      = write_output;
    pp_data.output_data.output_folder     = output_directory + "vtu/";
    pp_data.output_data.output_name       = output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.output_start_time = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.degree               = degree;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace RisingBubble
} // namespace FTI


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_ */
