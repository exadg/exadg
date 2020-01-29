/*
 * cavity_natural_convection.h
 *
 *  Created on: Jan 26, 2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// number of space dimensions
unsigned int const DIM = 2;

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

// set problem specific parameters like physical dimensions, etc.
double const L = 1.0; // Length of cavity
double const LEFT = 0.0;
double const RIGHT = 1.0;

double const START_TIME = 0.0;
double const END_TIME = 10.0;

double const CFL_OIF = 0.5;
double const CFL = CFL_OIF;
double const MAX_VELOCITY = 1.0;
bool const ADAPTIVE_TIME_STEPPING = true;

double const T_REF = 300.0;
double const DELTA_T = 20.0;
double const G = 10.0;

double const KINEMATIC_VISCOSITY = 1.0e-4;
double const PRANDTL = 1.0;
double const THERMAL_DIFFUSIVITY = KINEMATIC_VISCOSITY / PRANDTL;

// output
bool const WRITE_OUTPUT = true;
std::string const OUTPUT_FOLDER = "output_flow_with_transport/natural_convection/";
std::string const OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "test";
double const OUTPUT_START_TIME = START_TIME;
double const OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/100.0;

// restart
bool const WRITE_RESTART = false;
double const RESTART_INTERVAL_TIME = 10.0;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = DIM;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.right_hand_side = true;
  param.boussinesq_term = true;


  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.viscosity = KINEMATIC_VISCOSITY;
  param.reference_temperature = T_REF;
  param.thermal_expansion_coefficient = 1.0/param.reference_temperature;
  param.gravitational_force[0] = 0.0;
  param.gravitational_force[1] = -G;
  param.gravitational_force[2] = 0.0;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  param.adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.max_velocity = MAX_VELOCITY;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.cfl_oif = CFL_OIF;
  param.cfl = CFL;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // restart
  param.restart_data.write_restart = WRITE_RESTART;
  param.restart_data.interval_time = RESTART_INTERVAL_TIME;
  param.restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_fluid";

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Affine;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  param.continuity_penalty_use_boundary_data = true;
  param.apply_penalty_terms_in_postprocessing_step = true;
  param.type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;
  param.penalty_parameter_include_buoyancy_term = true;
  param.characteristic_velocity_buoyancy_term = 20.0*std::sqrt(G*RIGHT);


  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulation
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  // use FGMRES for matrix-free BlockJacobi or Multigrid with Krylov methods as smoother/coarse grid solver
  param.solver_momentum = SolverMomentum::FGMRES;
  param.solver_data_momentum = SolverData(1e4,1.e-12,1.e-6,100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;
  param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

  // preconditioner linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
}

}

namespace ConvDiff
{
void set_input_parameters(InputParameters &param, unsigned int const scalar_index)
{
  // MATHEMATICAL MODEL
  param.dim = DIM;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::ConvectionDiffusion;
  param.analytical_velocity_field = false;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = KINEMATIC_VISCOSITY;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.time_step_size = 1.0e-2;
  param.cfl = CFL;
  param.max_velocity = MAX_VELOCITY;
  param.exponent_fe_degree_convection = 1.5;
  param.exponent_fe_degree_diffusion = 3.0;
  param.diffusion_number = 0.01;
  param.dt_refinements = 0;

  // restart
  param.restart_data.write_restart = WRITE_RESTART;
  param.restart_data.interval_time = RESTART_INTERVAL_TIME;
  param.restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // SPATIAL DISCRETIZATION

  // triangulation
  param.triangulation_type = TriangulationType::Distributed;

  // polynomial degree
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Affine;

  // h-refinements
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  param.IP_factor = 1.0;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;

  // SOLVER
  param.solver = Solver::CG;
  param.solver_data = SolverData(1e4, 1.e-12, 1.e-6, 100);
  param.preconditioner = Preconditioner::InverseMassMatrix;
  param.update_preconditioner = false;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
}
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
  (void)periodic_faces;

  const double left = 0.0, right = L;
  GridGenerator::hyper_cube(*triangulation,left,right);
  triangulation->refine_global(n_refine_space);

  // set boundary IDs: 0 by default, set left boundary to 1
  typename Triangulation<dim>::cell_iterator cell;
  for(cell = triangulation->begin(); cell != triangulation->end(); ++cell)
  {
    for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12))
      {
         cell->face(face_number)->set_boundary_id(1);
      }
    }
  }
}

namespace IncNS
{

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param)
{
  (void)param;

  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = WRITE_OUTPUT;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME + "_fluid";
  pp_data.output_data.output_start_time = OUTPUT_START_TIME;
  pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.degree = param.degree_u;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

namespace ConvDiff
{

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0,new Functions::ConstantFunction<dim>(T_REF)));
  boundary_descriptor->dirichlet_bc.insert(pair(1,new Functions::ConstantFunction<dim>(T_REF+DELTA_T)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  field_functions->initial_solution.reset(new Functions::ConstantFunction<dim>(T_REF));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param,
                        unsigned int const              scalar_index)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = WRITE_OUTPUT;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
  pp_data.output_data.output_start_time = OUTPUT_START_TIME;
  pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  pp_data.output_data.degree = param.degree;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
