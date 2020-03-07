#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../grid_tools/dealii_extensions.h"
#include "../grid_tools/mesh_movement_functions.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;


// set problem specific parameters like physical dimensions, etc.
double const U_X_MAX   = 1.0;
double const VISCOSITY = 2.5e-2;

double const LEFT  = -0.5;
double const RIGHT =  0.5;

double const START_TIME = 0.0;
double const END_TIME   = 10.0;

bool const ALE = true;

namespace IncNS
{
void
set_input_parameters(InputParameters & param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.right_hand_side = false;

  // ALE
  param.ale_formulation = ALE;
  param.neumann_with_variable_normal_vector = false; // no Neumann boundaries for this test case

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time   = END_TIME;
  param.viscosity  = VISCOSITY;

  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.order_time_integrator = 1;
  param.start_with_low_order = false;
  param.adaptive_time_stepping = true;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.time_step_size = 0.25;
  param.dt_refinements = REFINE_TIME_MIN;
  param.max_velocity = std::sqrt(U_X_MAX*U_X_MAX + U_X_MAX*U_X_MAX);
  param.cfl = 0.25;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.c_eff = 8.0;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  param.cfl_oif = param.cfl/1.0;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = param.end_time - param.start_time;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  param.upwind_factor = 1.0;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // velocity pressure coupling terms
  param.gradp_formulation = FormulationPressureGradientTerm::Strong; //TODO //Weak;
  param.divu_formulation = FormulationVelocityDivergenceTerm::Strong; //TODO //Weak;

  // special case: pure DBC's
  param.pure_dirichlet_bc     = true;
  param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalSolutionInPoint;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  param.continuity_penalty_use_boundary_data = true;
  if(param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    param.apply_penalty_terms_in_postprocessing_step = false;
  else
    param.apply_penalty_terms_in_postprocessing_step = true;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;
  param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

  // PROJECTION METHODS
  param.store_previous_boundary_values = true;

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-14,1.e-14,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-14, 1.e-14);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  param.preconditioner_block_diagonal_projection = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal_projection = SolverData(1000,1.e-12,1.e-2,1000);

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator<=2 ? param.order_time_integrator : 2;
  param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-14,1.e-14);
  param.preconditioner_viscous = PreconditionerViscous::Multigrid;
  param.multigrid_data_viscous.type = MultigridType::hMG;
  param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.update_preconditioner_viscous = false;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = std::min(2, (int)param.order_time_integrator) - 1;  // J_p = J-1, but not larger than 1
  param.rotational_formulation       = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-12);

  // linear solver
  param.solver_momentum = SolverMomentum::FGMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-14, 1.e-14, 100);
  param.update_preconditioner_momentum = false;
  param.preconditioner_momentum = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;

  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-12);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-14, 100);

  // preconditioner linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = false;

  // preconditioner momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data_velocity_block.type = MultigridType::phMG;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  // coarse grid solver
  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev; //GMRES;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
}
} // namespace IncNS


/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
  unsigned int const                            n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    /*periodic_faces*/)
{
  const double left = LEFT, right = RIGHT;
  GridGenerator::hyper_cube(*triangulation, left, right);

  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                               MESH MOTION                                                */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
std::shared_ptr<Function<dim>>
set_mesh_movement_function()
{
  std::shared_ptr<Function<dim>> mesh_motion;

  MeshMovementData<dim> data;
  data.temporal = MeshMovementAdvanceInTime::Sin;
  data.shape = MeshMovementShape::Sin;
  data.dimensions[0] = std::abs(RIGHT-LEFT);
  data.dimensions[1] = std::abs(RIGHT-LEFT);
  data.amplitude = 0.08 * (RIGHT-LEFT);
  data.period = (END_TIME-START_TIME)/10.0;
  data.t_start = START_TIME;
  data.t_end = END_TIME;
  data.spatial_number_of_oscillations = 1.0;
  mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

  return mesh_motion;
}


namespace IncNS
{

/**************************************************************************************/
/*                                                                                    */
/*          FUNCTIONS (ANALYTICAL/INITIAL SOLUTION, BOUNDARY CONDITIONS, etc.)        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
  {
    double result = 1.0;
    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(const double time = 0.) : Function<dim>(1 /*n_components*/, time)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
  {
    double result = 1.0;
    return result;
  }
};


template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                        std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new AnalyticalSolutionVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number>>
construct_postprocessor(InputParameters const & param, MPI_Comm const &mpi_comm)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output                    = false;
  pp_data.output_data.output_folder                   = "output/free_stream_preservation/vtu/";
  pp_data.output_data.output_name                     = "test";
  pp_data.output_data.output_start_time               = param.start_time;
  pp_data.output_data.output_interval_time            = (param.end_time - param.start_time)/100;
  pp_data.output_data.write_higher_order              = true;
  pp_data.output_data.degree                          = param.degree_u;

  // calculation of velocity error
  pp_data.error_data_u.analytical_solution_available = true;
  pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
  pp_data.error_data_u.calculate_relative_errors = true;
  pp_data.error_data_u.error_calc_start_time     = param.start_time;
  pp_data.error_data_u.error_calc_interval_time  = (param.end_time - param.start_time)/10;
  pp_data.error_data_u.name                      = "velocity";

  // ... pressure error
  pp_data.error_data_p.analytical_solution_available = true;
  pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
  pp_data.error_data_p.calculate_relative_errors = true;
  pp_data.error_data_p.error_calc_start_time     = param.start_time;
  pp_data.error_data_p.error_calc_interval_time  = (param.end_time - param.start_time)/10;
  pp_data.error_data_p.name                      = "pressure";

  std::shared_ptr<PostProcessorBase<dim, Number>> pp;
  pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

  return pp;
}

} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_ */
