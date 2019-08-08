#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_
#  define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_VORTEX_H_

#  include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#  include "../grid_tools/dealii_extensions.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 5;
unsigned int const DEGREE_MAX = 5;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;


// set problem specific parameters like physical dimensions, etc.
const double                 U_X_MAX                  = 1.0;
const double                 VISCOSITY                = 2.5e-2; // 1.e-2; //2.5e-2;
const FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::LaplaceFormulation;

enum class MeshType
{
  UniformCartesian,
  ComplexSurfaceManifold,
  ComplexVolumeManifold,
  Curvilinear
};
const MeshType MESH_TYPE = MeshType::UniformCartesian;

const AnalyicMeshMovement MESH_MOVEMENT = AnalyicMeshMovement::CubeDoubleInteriorSinCos;
const bool INITIALIZE_WITH_FORMER_MESH_INSTANCES = false;
const double TRIANGULATION_LEFT               = -0.5;
const double TRIANGULATION_RIGHT              = 0.5;
const double TRIANGULATION_MOVEMENT_AMPLITUDE = 0.04;
const double TRIANGULATION_MOVEMENT_FREQUENCY = 0.8;

const double START_TIME = 0.0;
const double END_TIME   = 0.5;

bool PURE_DIRICHLET = true;

namespace IncNS
{
void
set_input_parameters(InputParameters & param)
{
  // ALE
  param.grid_velocity_analytical                 = false;
  param.ale_formulation                          = true;
  param.NBC_prescribed_with_known_normal_vectors = true;
  param.initialize_with_former_mesh_instances    = INITIALIZE_WITH_FORMER_MESH_INSTANCES;
  param.start_with_low_order                     = true;
  param.time_step_size                           = 5e-5; // 0.5;//5e-5;
  param.order_time_integrator                    = 1;
  param.temporal_discretization                  = TemporalDiscretization::BDFCoupledSolution;
  param.calculation_of_time_step_size            = TimeStepCalculation::CFL;
  param.adaptive_time_stepping                   = true;
  param.cfl                                      = 0.4;

  // MATHEMATICAL MODEL
  param.dim                         = 2;
  param.problem_type                = ProblemType::Unsteady;
  param.equation_type               = EquationType::NavierStokes;
  param.formulation_viscous_term    = FORMULATION_VISCOUS_TERM;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.right_hand_side             = false;



  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time   = END_TIME;
  param.viscosity  = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;


  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.time_integrator_oif          = TimeIntegratorOIF::ExplRK3Stage7Reg2;

  param.max_velocity                    = U_X_MAX;
  param.cfl_oif                         = param.cfl / 1.0;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.c_eff                           = 8.0;


  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen    = true;
  param.solver_info_data.interval_wall_time = 2 * 3600;

  // restart
  param.restarted_simulation             = false;
  param.restart_data.write_restart       = false;
  param.restart_data.interval_time       = 0.25;
  param.restart_data.interval_wall_time  = 1.e6;
  param.restart_data.interval_time_steps = 1e8;
  param.restart_data.filename = "output/free_stream_preservation/free_stream_preservation";


  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u           = DEGREE_MIN;
  param.degree_p           = DegreePressure::MixedOrder;
  param.mapping            = MappingType::Isoparametric;
  param.h_refinements      = REFINE_SPACE_MIN;

  // convective term
  param.upwind_factor = 1.0;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc     = PURE_DIRICHLET;
  param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalSolutionInPoint;


  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = true;
  param.use_cell_based_face_loops                           = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson         = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-6, 100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
    PreconditionerSmoother::PointJacobi;

  // projection step
  param.solver_projection                        = SolverProjection::CG;
  param.solver_data_projection                   = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_projection                = PreconditionerProjection::InverseMassMatrix;
  param.preconditioner_block_diagonal_projection = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal_projection    = SolverData(1000, 1.e-12, 1.e-2, 1000);

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc =
    param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous                = SolverViscous::CG;
  param.solver_data_viscous           = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_viscous        = PreconditionerViscous::InverseMassMatrix;
  param.update_preconditioner_viscous = false;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = param.order_time_integrator - 1;
  param.rotational_formulation       = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100, 1.e-12, 1.e-6);

  // linear solver
  param.solver_momentum                  = SolverMomentum::FGMRES;
  param.solver_data_momentum             = SolverData(1e4, 1.e-12, 1.e-6, 100);
  param.preconditioner_momentum          = MomentumPreconditioner::InverseMassMatrix;
  param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  param.update_preconditioner_momentum                 = true;

  // Jacobi smoother data
  param.multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_momentum.smoother_data.iterations     = 5;
  param.multigrid_data_momentum.coarse_problem.solver        = MultigridCoarseGridSolver::GMRES;

  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100, 1.e-12, 1.e-6);

  // linear solver
  param.solver_coupled      = SolverCoupled::FGMRES; // FGMRES; //GMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // preconditioner linear solver
  param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = true;

  // preconditioner momentum block
  param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data_velocity_block.type     = MultigridType::phMG;
  param.multigrid_data_velocity_block.smoother_data.smoother =
    MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner =
    PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  // coarse grid solver
  param.multigrid_data_velocity_block.coarse_problem.solver =
    MultigridCoarseGridSolver::Chebyshev; // GMRES;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;

  param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
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
  std::shared_ptr<parallel::Triangulation<dim>> triangulation,
  unsigned int const                            n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_faces)
{
  const double left = TRIANGULATION_LEFT, right = TRIANGULATION_RIGHT;
  GridGenerator::hyper_cube(*triangulation, left, right);

  // use periodic boundary conditions
  // x-direction
  triangulation->begin()->face(0)->set_all_boundary_ids(0 + 10);
  triangulation->begin()->face(1)->set_all_boundary_ids(1 + 10);
  // y-direction
  triangulation->begin()->face(2)->set_all_boundary_ids(2 + 10);
  triangulation->begin()->face(3)->set_all_boundary_ids(3 + 10);

  auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0 + 10, 1 + 10, 0, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2 + 10, 3 + 10, 1, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  triangulation->refine_global(n_refine_space);
}

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


namespace IncNS
{
template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                        std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new AnalyticalSolutionVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new AnalyticalSolutionPressure<dim>()));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  MeshMovementData data;
  data.type = MESH_MOVEMENT;
  data.left = TRIANGULATION_LEFT;
  data.right = TRIANGULATION_RIGHT;
  data.A = TRIANGULATION_MOVEMENT_AMPLITUDE;
  data.f = TRIANGULATION_MOVEMENT_FREQUENCY;
  data.t_0 = START_TIME;
  data.t_end = END_TIME;
  data.initialize_with_former_mesh_instances = INITIALIZE_WITH_FORMER_MESH_INSTANCES;

  if(data.type == AnalyicMeshMovement::CubeInteriorSinCos)
    field_functions->analytical_solution_grid_velocity.reset(new CubeInteriorSinCos<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeInteriorSinCosOnlyX)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeInteriorSinCosOnlyX<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeInteriorSinCosOnlyY)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeInteriorSinCosOnlyY<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeSinCosWithBoundaries)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeSinCosWithBoundaries<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeInteriorSinCosWithSinInTime)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeInteriorSinCosWithSinInTime<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeXSquaredWithBoundaries)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeXSquaredWithBoundaries<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeDoubleInteriorSinCos)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeDoubleInteriorSinCos<dim>(data));
  else if(data.type == AnalyicMeshMovement::CubeDoubleSinCosWithBoundaries)
    field_functions->analytical_solution_grid_velocity.reset(
      new CubeDoubleSinCosWithBoundaries<dim>(data));
  else
    AssertThrow(false, ExcMessage("No suitable mesh movement for test case defined!"));

  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number>>
construct_postprocessor(InputParameters const & param)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output                    = true;
  pp_data.output_data.output_folder                   = "output/free_stream_preservation/vtu/";
  pp_data.output_data.output_name                     = "free_stream_preservation";
  pp_data.output_data.output_start_time               = param.start_time;
  pp_data.output_data.output_interval_time            = (param.end_time - param.start_time) / 20;
  pp_data.output_data.write_vorticity                 = true;
  pp_data.output_data.write_divergence                = true;
  pp_data.output_data.write_velocity_magnitude        = true;
  pp_data.output_data.write_vorticity_magnitude       = true;
  pp_data.output_data.write_processor_id              = true;
  pp_data.output_data.mean_velocity.calculate         = false;
  pp_data.output_data.mean_velocity.sample_start_time = param.start_time;
  pp_data.output_data.mean_velocity.sample_end_time   = param.end_time;
  pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
  pp_data.output_data.write_higher_order                   = true;
  pp_data.output_data.degree                               = param.degree_u;

  // calculation of velocity error
  pp_data.error_data_u.analytical_solution_available = true;
  pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
  pp_data.error_data_u.calculate_relative_errors = true;
  pp_data.error_data_u.error_calc_start_time     = param.start_time;
  pp_data.error_data_u.error_calc_interval_time  = (param.end_time - param.start_time);
  pp_data.error_data_u.name                      = "velocity";

  // ... pressure error
  pp_data.error_data_p.analytical_solution_available = true;
  pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
  pp_data.error_data_p.calculate_relative_errors = true;
  pp_data.error_data_p.error_calc_start_time     = param.start_time;
  pp_data.error_data_p.error_calc_interval_time  = (param.end_time - param.start_time);
  pp_data.error_data_p.name                      = "pressure";

  std::shared_ptr<PostProcessorBase<dim, Number>> pp;
  pp.reset(new PostProcessor<dim, Number>(pp_data));

  return pp;
}

} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_ALE_TEST_CASES_FREE_STREAM_PRESERVATION_H_ */
