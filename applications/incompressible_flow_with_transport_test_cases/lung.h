/*
 * 3D_taylor_green_vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include "../grid_tools/lung/lung_environment.h"
#include "../grid_tools/lung/lung_grid.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 0;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// number of lung generations
unsigned int const GENERATIONS = 3;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.7e-5; //1.7e-5; // m^2/s
const double MAX_VELOCITY = 3.0; //20.0; // m/s
const double PERIOD = 0.1; // 100 ms
double const END_TIME = PERIOD;// TODO // * 8.0;
double const PEEP = 3.0; //TODO //8.0 * 98.0665; // 8 cmH20, 1 cmH20 = 98.0665 Pa

std::string OUTPUT_FOLDER = "output/lung/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test_geoemtry";

// solver tolerances
const double ABS_TOL = 1.e-12;
const double REL_TOL = 1.e-3;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  use_outflow_bc_convective_term = true;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  time_integrator_oif = TimeIntegratorOIF::ExplRK2Stage2;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = MAX_VELOCITY;
  cfl_oif = 0.4;
  cfl = cfl_oif * 1.0;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-3;
  order_time_integrator = 2;
  start_with_low_order = true;

  // NUMERICAL PARAMETERS
  implement_block_diagonal_preconditioner_matrix_free = false;
  use_cell_based_face_loops = false;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::FullyDistributed;

  // mapping
  degree_mapping = 1; //FE_DEGREE_VELOCITY; //TODO

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's (only periodic BCs -> pure_dirichlet_bc = true)
  pure_dirichlet_bc = false;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_factor = divergence_penalty_factor;
  add_penalty_terms_to_monolithic_system = false;

  // TURBULENCE
  use_turbulence_model = false;
  turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165
  // Vreman: 0.28
  // WALE: 0.50
  // Sigma: 1.35
  turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_data_pressure_poisson = SolverData(1000,ABS_TOL,REL_TOL,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.type = MultigridType::phMG;
  multigrid_data_pressure_poisson.p_sequence = PSequenceType::Bisect;
  multigrid_data_pressure_poisson.dg_to_cg_transfer = DG_To_CG_Transfer::Fine;
  multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG; //Chebyshev;
  multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG; //PointJacobi;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, ABS_TOL, REL_TOL);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  update_preconditioner_projection = false;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,ABS_TOL,REL_TOL);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
//  preconditioner_viscous = PreconditionerViscous::Multigrid;
  multigrid_data_viscous.type = MultigridType::hpMG;
  multigrid_data_viscous.p_sequence = PSequenceType::Bisect;
  multigrid_data_viscous.dg_to_cg_transfer = DG_To_CG_Transfer::None;
  multigrid_data_viscous.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  multigrid_data_viscous.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  update_preconditioner_viscous = true;
  update_preconditioner_viscous_every_time_steps = 1;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  else
    solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);

  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = true;

  // formulation
  order_pressure_extrapolation = order_time_integrator-1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::GMRES;
  solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion; //CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  exact_inversion_of_laplace_operator = false;
  solver_data_pressure_block = SolverData(1e4, 1.e-12, 1.e-6, 100);


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = PERIOD/20;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.write_velocity_magnitude = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_q_criterion = true;
  output_data.write_processor_id = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;

  // calculate div and mass error
  mass_data.calculate_error = false;
  mass_data.start_time = 0.0;
  mass_data.sample_every_time_steps = 1e2;
  mass_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
  mass_data.reference_length_scale = 1.0;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = PERIOD/20;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/



/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class VelocityInlet : public Function<dim>
{
public:
  VelocityInlet (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  component = 0) const
  {
    if(component == 2)
      return -1.0;
    else
      return 1.0;
  }
};

template<int dim>
class PressureInlet : public Function<dim>
{
public:
  PressureInlet (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &/*p*/,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    double pressure = 0.0;

    if((int(t/(PERIOD/2)))%2 == 0)
    {
      pressure = PEEP;
    }
    else
    {
      pressure = -PEEP;
    }

    return pressure;
  }
};

template<int dim>
void create_grid_and_set_boundary_conditions(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  std::vector<std::string> files;
  get_lung_files_from_environment(files);
  auto tree_factory = dealii::GridGenerator::lung_files_to_node(files);

  std::map<std::string, double> timings;

  // create triangulation
  if(auto tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
  {
    dealii::GridGenerator::lung(
      *tria, GENERATIONS, n_refine_space, n_refine_space, tree_factory, timings);
  }
  else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
  {
    dealii::GridGenerator::lung(*tria, GENERATIONS, n_refine_space, tree_factory, timings);
  }
  else
  {
    AssertThrow(false, ExcMessage("Unknown triangulation!"));
  }

  // set boundary conditions
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // 0 = walls
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

  // 1 = inlet
  boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new PressureInlet<dim>()));

//  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1, new VelocityInlet<dim>()));
//  boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

  // 2 = outlet
  boundary_descriptor_velocity->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}

// Postprocessor
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
std::shared_ptr<PostProcessorBase<dim, fe_degree_u, fe_degree_p, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.kinetic_energy_spectrum_data = param.kinetic_energy_spectrum_data;

  std::shared_ptr<PostProcessor<dim,fe_degree_u,fe_degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,fe_degree_u,fe_degree_p,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
