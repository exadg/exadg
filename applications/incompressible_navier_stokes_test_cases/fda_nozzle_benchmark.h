/*
 * fda_nozzle_benchmark.h
 *
 *  Created on: May, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

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

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for DOMAIN 1
unsigned int const REFINE_STEPS_SPACE_DOMAIN1 = 4;

// set the number of refine levels for DOMAIN 2
unsigned int const REFINE_STEPS_SPACE_DOMAIN2 = 3;

// needed for single domain solver only
unsigned int const REFINE_STEPS_SPACE_MIN = REFINE_STEPS_SPACE_DOMAIN2;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_DOMAIN2;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// prescribe velocity inflow profile for nozzle domain via precursor simulation?
// USE_PRECURSOR_SIMULATION == true:  use solver unsteady_navier_stokes_two_domains.cc
// USE_PRECURSOR_SIMULATION == false: use solver unsteady_navier_stokes.cc
bool const USE_PRECURSOR_SIMULATION = true;

// use prescribed velocity profile at inflow superimposed by random perturbations (white noise)?
// This option is only relevant if USE_PRECURSOR_SIMULATION == false
bool const USE_RANDOM_PERTURBATION = false;
// amplitude of perturbations relative to maximum velocity on centerline
double const FACTOR_RANDOM_PERTURBATIONS = 0.05;

// set the throat Reynolds number Re_throat = U_{mean,throat} * (2 R_throat) / nu
double const RE = 8000; //500; //2000; //3500; //5000; //6500; //8000;

// output folders
std::string OUTPUT_FOLDER = "output/fda/paper_final/Re8000/l4_l3_k32/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_1 = "precursor";
std::string OUTPUT_NAME_2 = "nozzle";
std::string FILENAME_FLOWRATE = "precursor_mean_velocity";

// set problem specific parameters like physical dimensions, etc.

// radius
double const R = 0.002;
double const R_INNER = R;
double const R_OUTER = 3.0*R;
double const D = 2.0*R_OUTER;

// lengths (dimensions in flow direction z)
double const LENGTH_PRECURSOR = 8.0*R_OUTER;
double const LENGTH_INFLOW = 8.0*R_OUTER;
double const LENGTH_CONE = (R_OUTER-R_INNER)/std::tan(20.0/2.0*numbers::PI/180.0);
double const LENGTH_THROAT = 0.04;
double const LENGTH_OUTFLOW = 20.0*R_OUTER;
double const OFFSET = 2.0*R_OUTER;

// mesh parameters
unsigned int const N_CELLS_AXIAL = 2;
unsigned int const N_CELLS_AXIAL_PRECURSOR = 4*N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_INFLOW = 4*N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_CONE = 2*N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_THROAT = 4*N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_OUTFLOW = 10*N_CELLS_AXIAL;

unsigned int const MANIFOLD_ID_CYLINDER = 1234;
unsigned int const MANIFOLD_ID_OFFSET_CONE = 7890;

// z-coordinates
double const Z2_OUTFLOW = LENGTH_OUTFLOW;
double const Z1_OUTFLOW = 0.0;

double const Z2_THROAT = 0.0;
double const Z1_THROAT = - LENGTH_THROAT;

double const Z2_CONE = - LENGTH_THROAT;
double const Z1_CONE = - LENGTH_THROAT - LENGTH_CONE;

double const Z2_INFLOW = - LENGTH_THROAT - LENGTH_CONE;
double const Z1_INFLOW = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW;

double const Z2_PRECURSOR = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET;
double const Z1_PRECURSOR = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET - LENGTH_PRECURSOR;

double const AREA_INFLOW = R_OUTER*R_OUTER*numbers::PI;
double const AREA_THROAT = R_INNER*R_INNER*numbers::PI;

// kinematic viscosity (same viscosity for all Reynolds numbers)
double const VISCOSITY = 3.31e-6;

double const MEAN_VELOCITY_THROAT = RE * VISCOSITY / (2.0*R_INNER);
double const TARGET_FLOW_RATE = MEAN_VELOCITY_THROAT*AREA_THROAT;
double const MEAN_VELOCITY_INFLOW = TARGET_FLOW_RATE/AREA_INFLOW;

double const MAX_VELOCITY = 2.0*TARGET_FLOW_RATE/AREA_INFLOW;
double const MAX_VELOCITY_CFL = 2.0*TARGET_FLOW_RATE/AREA_THROAT;

// start and end time

// estimation of flow-through time T_0 (through nozzle section)
// based on the mean velocity through throat
double const T_0 = LENGTH_THROAT/MEAN_VELOCITY_THROAT;
double const START_TIME_PRECURSOR = -500.0*T_0; // let the flow develop
double const START_TIME_NOZZLE = 0.0*T_0;
double const END_TIME = 250.0*T_0; //150.0*T_0;

// output
bool const WRITE_OUTPUT = true;
double const OUTPUT_START_TIME_PRECURSOR = START_TIME_PRECURSOR;
double const OUTPUT_START_TIME_NOZZLE = START_TIME_NOZZLE;
double const OUTPUT_INTERVAL_TIME = 5.0*T_0;  //10.0*T_0;

// sampling

// sampling interval should last over (100-200) * T_0 according to preliminary results.
// might be reduced when using averaging in circumferential direction.
double const SAMPLE_START_TIME = 50.0*T_0; // let the flow develop
double const SAMPLE_END_TIME = END_TIME; // that's the only reasonable choice
unsigned int SAMPLE_EVERY_TIMESTEPS = 1;
unsigned int WRITE_OUTPUT_EVERY_TIMESTEPS = SAMPLE_EVERY_TIMESTEPS*100;

// line plot data
unsigned int N_POINTS_LINE_AXIAL = 400;
unsigned int N_POINTS_LINE_RADIAL = 64;
unsigned int N_POINTS_LINE_CIRCUMFERENTIAL = 32;
QuantityStatistics<DIMENSION> QUANTITY_VELOCITY;
QuantityStatistics<DIMENSION> QUANTITY_VELOCITY_CIRCUMFERENTIAL;

// data structures that we need to apply the velocity inflow profile:

// - we currently use global variables for this purpose
// - choose a large number of points to ensure a smooth inflow profile
unsigned int N_POINTS_R = 10 * (FE_DEGREE_VELOCITY+1) * std::pow(2.0, REFINE_STEPS_SPACE_DOMAIN1); //100;
unsigned int N_POINTS_PHI = N_POINTS_R;
std::vector<double> R_VALUES(N_POINTS_R);
std::vector<double> PHI_VALUES(N_POINTS_PHI);
std::vector<Tensor<1,DIMENSION,double> > VELOCITY_VALUES(N_POINTS_R*N_POINTS_PHI);

// data structures that we need to control the mass flow rate:
// NOTA BENE: these variables will be modified by the postprocessor!
double FLOW_RATE = 0.0;
// the flow rate controller also needs the time step size as parameter
double TIME_STEP_FLOW_RATE_CONTROLLER = 1.0;

class FlowRateController
{
public:
  FlowRateController()
    :
    // initialize the body force such that the desired flow rate is obtained
    // under the assumption of a parabolic velocity profile in radial direction
    f(4.0*VISCOSITY*MAX_VELOCITY/std::pow(R_OUTER,2.0)) // f(t=t_0) = f_0
  {}

  double get_body_force()
  {
    return f;
  }

  void update_body_force()
  {
    // use an I-controller to asymptotically reach the desired target flow rate

    // dimensional analysis: [k] = 1/(m^2 s^2) -> k = const * U_{mean,inflow}^2 / D^4
    // constant: choose a default value of 1
    double const k = 1.0e0*std::pow(MEAN_VELOCITY_INFLOW,2.0)/std::pow(D,4.0);
    f += k*(TARGET_FLOW_RATE - FLOW_RATE)*TIME_STEP_FLOW_RATE_CONTROLLER;
  }

private:
  double f;
};

// use a global variable which will be called by the postprocessor
// in order to update the body force.
FlowRateController FLOW_RATE_CONTROLLER;

// initialize vectors
void initialize_r_and_phi_values()
{
  AssertThrow(N_POINTS_R >= 2, ExcMessage("Variable N_POINTS_R is invalid"));
  AssertThrow(N_POINTS_PHI >= 2, ExcMessage("Variable N_POINTS_PHI is invalid"));

  for(unsigned int i=0; i<N_POINTS_R; ++i)
    R_VALUES[i] = double(i)/double(N_POINTS_R-1)*R_OUTER;

  for(unsigned int i=0; i<N_POINTS_PHI; ++i)
    PHI_VALUES[i] = -numbers::PI + double(i)/double(N_POINTS_PHI-1)*2.0*numbers::PI;
}

void initialize_velocity_values()
{
  AssertThrow(N_POINTS_R >= 2, ExcMessage("Variable N_POINTS_R is invalid"));
  AssertThrow(N_POINTS_PHI >= 2, ExcMessage("Variable N_POINTS_PHI is invalid"));

  for(unsigned int iy=0; iy<N_POINTS_R; ++iy)
  {
    for(unsigned int iz=0; iz<N_POINTS_PHI; ++iz)
    {
      Tensor<1,DIMENSION,double> velocity;
      // flow in z-direction
      velocity[2] = MAX_VELOCITY*(1.0-std::pow(R_VALUES[iy]/R_OUTER,2.0));
      
      if(USE_RANDOM_PERTURBATION==true)
      {
        // Add random perturbation
        double perturbation = FACTOR_RANDOM_PERTURBATIONS * velocity[2] * ((double)rand()/RAND_MAX-0.5)/0.5;
        velocity[2] += perturbation;
      }

      VELOCITY_VALUES[iy*N_POINTS_R + iz] = velocity;
    }
  }
}

/*
 *  This function returns the radius of the cross-section at a
 *  specified location z in streamwise direction.
 */
double radius_function(double const z)
{
  double radius = R_OUTER;

  if(z >= Z1_INFLOW && z <= Z2_INFLOW)
    radius = R_OUTER;
  else if(z >= Z1_CONE && z <= Z2_CONE)
    radius = R_OUTER * (1.0 - (z-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER);
  else if(z >= Z1_THROAT && z <= Z2_THROAT)
    radius = R_INNER;
  else if(z > Z1_OUTFLOW && z <= Z2_OUTFLOW)
    radius = R_OUTER;

  return radius;
}

/*
 *  To set input parameters for DOMAIN 1 and DOMAIN 2, use
 *
 *  if(domain_id == 1){}
 *  else if(domain_id == 2){}
 *
 *  Most of the input parameters are the same for both domains!
 *
 *  DOMAIN 1: precursor (used to generate inflow data)
 *  DOMAIN 2: nozzle (the actual domain of interest)
 */
template<int dim>
void InputParameters<dim>::set_input_parameters(unsigned int const domain_id)
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  use_outflow_bc_convective_term = true;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  if(domain_id == 1)
    start_time = START_TIME_PRECURSOR;
  else if(domain_id == 2)
    start_time = START_TIME_NOZZLE;

  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;

  // The pressure-correction scheme with an implicit treatment of the convective term
  // (using CFL number > 1) was found to be more efficient for this test case
  // than, e.g., the dual splitting scheme with explicit formulation of the convective term.
  
//  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
//  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
//  calculation_of_time_step_size = TimeStepCalculation::AdaptiveTimeStepCFL;
  temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  adaptive_time_stepping_limiting_factor = 3.0;
  max_velocity = MAX_VELOCITY_CFL;
  // ConstTimeStepCFL: CFL_critical = 0.3 - 0.5 for k=3
  // AdaptiveTimeStepCFL: CFL_critical = 0.125 - 0.15 for k=3
  // Best pratice: use CFL = 4.0 for implicit treatment (e.g., pressure-correction scheme)
  // and CFL = 0.13 with adaptive time stepping for an explicit treatment (e.g., dual splitting)
  cfl = 4.0;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;
  penalty_term_div_formulation = PenaltyTermDivergenceFormulation::Symmetrized;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  if(domain_id == 1)
    pure_dirichlet_bc = true;
  else if(domain_id == 2)
    pure_dirichlet_bc = false;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  continuity_penalty_use_boundary_data = false;
  type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;
  continuity_penalty_factor = divergence_penalty_factor;

  // TURBULENCE
  use_turbulence_model = false;
  turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165, Vreman: 0.28, WALE: 0.50, Sigma: 1.35
  turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;

  // Best practice: use PCG with Jacobi preconditioner for refine level l=0,
  // and FGMRES solver with Chebyshev smoother and PCG_PointJaocbi coarse grid solver for
  // refinement levels l=1 and larger.
  if(REFINE_STEPS_SPACE_MIN == 0)
  {
    solver_pressure_poisson = SolverPressurePoisson::PCG;
    preconditioner_pressure_poisson = PreconditionerPressurePoisson::Jacobi;
  }
  else
  {
    solver_pressure_poisson = SolverPressurePoisson::FGMRES;
    preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
    multigrid_data_pressure_poisson.smoother = MultigridSmoother::Chebyshev;
    multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::PCG_PointJacobi;
  }

  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-3;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix; //BlockJacobi; //PointJacobi; //InverseMassMatrix;
  update_preconditioner_projection = true;
  abs_tol_projection = 1.e-12;
  rel_tol_projection = 1.e-3;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-3;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
  rotational_formulation = true; // use false for standard formulation

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-3;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  abs_tol_momentum_linear = 1.e-12;
  if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  { rel_tol_momentum_linear = 1.e-1; }
  else{ rel_tol_momentum_linear = 1.e-3; }
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;
  scaling_factor_continuity = 1.0;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-3;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES; //GMRES; //FGMRES;
  abs_tol_linear = 1.e-12;
  if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  { rel_tol_linear = 1.e-1; }
  else{ rel_tol_linear = 1.e-3; }
  max_iter_linear = 1e3;
  max_n_tmp_vectors = 100;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;
  update_preconditioner = false;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // Chebyshev moother
  multigrid_data_schur_complement_preconditioner.smoother = MultigridSmoother::Chebyshev;
  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::Chebyshev;


  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;

  // output of solver information
  output_solver_info_every_timesteps = 1e3; //1e5;

  if(domain_id == 1)
  {
    // write output for visualization of results
    output_data.write_output = WRITE_OUTPUT;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_1;
    output_data.output_start_time = OUTPUT_START_TIME_PRECURSOR;
    output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
    output_data.write_divergence = true;
    output_data.write_processor_id = true;
    output_data.mean_velocity.calculate = true;
    output_data.mean_velocity.sample_start_time = SAMPLE_START_TIME;
    output_data.mean_velocity.sample_end_time = SAMPLE_END_TIME;
    output_data.mean_velocity.sample_every_timesteps = 1;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // inflow data
    // prescribe solution at the right boundary of the precursor domain
    // as weak Dirichlet boundary condition at the left boundary of the nozzle domain
    inflow_data.write_inflow_data = true;
    inflow_data.inflow_geometry = InflowGeometry::Cylindrical;
    inflow_data.normal_direction = 2;
    inflow_data.normal_coordinate = Z2_PRECURSOR;
    inflow_data.n_points_y = N_POINTS_R;
    inflow_data.n_points_z = N_POINTS_PHI;
    inflow_data.y_values = &R_VALUES;
    inflow_data.z_values = &PHI_VALUES;
    inflow_data.array = &VELOCITY_VALUES;

    // calculation of flow rate (use volume-based computation)
    mean_velocity_data.calculate = true;
    mean_velocity_data.filename_prefix = OUTPUT_FOLDER + FILENAME_FLOWRATE;
    Tensor<1,dim,double> direction; direction[2] = 1.0;
    mean_velocity_data.direction = direction;
    mean_velocity_data.write_to_file = true;
  }
  else if(domain_id == 2)
  {
    // write output for visualization of results
    output_data.write_output = WRITE_OUTPUT;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_2;
    output_data.output_start_time = OUTPUT_START_TIME_NOZZLE;
    output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
    output_data.write_divergence = true;
    output_data.write_processor_id = true;
    output_data.mean_velocity.calculate = true;
    output_data.mean_velocity.sample_start_time = SAMPLE_START_TIME;
    output_data.mean_velocity.sample_end_time = SAMPLE_END_TIME;
    output_data.mean_velocity.sample_every_timesteps = 1;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // evaluation of quantities along lines
    line_plot_data.write_output = true;
    line_plot_data.filename_prefix = OUTPUT_FOLDER;
    line_plot_data.statistics_data.calculate_statistics = true;
    line_plot_data.statistics_data.sample_start_time = SAMPLE_START_TIME;
    line_plot_data.statistics_data.sample_end_time = END_TIME;
    line_plot_data.statistics_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
    line_plot_data.statistics_data.write_output_every_timesteps = WRITE_OUTPUT_EVERY_TIMESTEPS;

    // lines
    Line<dim> axial_profile, radial_profile_z1, radial_profile_z2, radial_profile_z3, radial_profile_z4,
              radial_profile_z5, radial_profile_z6, radial_profile_z7, radial_profile_z8, radial_profile_z9,
              radial_profile_z10, radial_profile_z11, radial_profile_z12;

    double z_1 = -0.088, z_2 = - 0.064, z_3 = -0.048, z_4 = -0.02, z_5 = -0.008, z_6 = 0.0,
           z_7 = 0.008, z_8 = 0.016, z_9 = 0.024, z_10 = 0.032, z_11 = 0.06, z_12 = 0.08;

    // begin and end points of all lines
    axial_profile.begin =      Point<dim> (0,0,Z1_INFLOW);
    axial_profile.end =        Point<dim> (0,0,Z2_OUTFLOW);
    radial_profile_z1.begin =  Point<dim> (0,0,z_1);
    radial_profile_z1.end =    Point<dim> (radius_function(z_1),0,z_1);
    radial_profile_z2.begin =  Point<dim> (0,0,z_2);
    radial_profile_z2.end =    Point<dim> (radius_function(z_2),0,z_2);
    radial_profile_z3.begin =  Point<dim> (0,0,z_3);
    radial_profile_z3.end =    Point<dim> (radius_function(z_3),0,z_3);
    radial_profile_z4.begin =  Point<dim> (0,0,z_4);
    radial_profile_z4.end =    Point<dim> (radius_function(z_4),0,z_4);
    radial_profile_z5.begin =  Point<dim> (0,0,z_5);
    radial_profile_z5.end =    Point<dim> (radius_function(z_5),0,z_5);
    radial_profile_z6.begin =  Point<dim> (0,0,z_6);
    radial_profile_z6.end =    Point<dim> (radius_function(z_6),0,z_6);
    radial_profile_z7.begin =  Point<dim> (0,0,z_7);
    radial_profile_z7.end =    Point<dim> (radius_function(z_7),0,z_7);
    radial_profile_z8.begin =  Point<dim> (0,0,z_8);
    radial_profile_z8.end =    Point<dim> (radius_function(z_8),0,z_8);
    radial_profile_z9.begin =  Point<dim> (0,0,z_9);
    radial_profile_z9.end =    Point<dim> (radius_function(z_9),0,z_9);
    radial_profile_z10.begin = Point<dim> (0,0,z_10);
    radial_profile_z10.end =   Point<dim> (radius_function(z_10),0,z_10);
    radial_profile_z11.begin = Point<dim> (0,0,z_11);
    radial_profile_z11.end =   Point<dim> (radius_function(z_11),0,z_11);
    radial_profile_z12.begin = Point<dim> (0,0,z_12);
    radial_profile_z12.end =   Point<dim> (radius_function(z_12),0,z_12);

    // number of points
    axial_profile.n_points =      N_POINTS_LINE_AXIAL;
    radial_profile_z1.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z2.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z3.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z4.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z5.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z6.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z7.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z8.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z9.n_points =  N_POINTS_LINE_RADIAL;
    radial_profile_z10.n_points = N_POINTS_LINE_RADIAL;
    radial_profile_z11.n_points = N_POINTS_LINE_RADIAL;
    radial_profile_z12.n_points = N_POINTS_LINE_RADIAL;

    // quantities

    // no additional averaging in space for centerline velocity
    QUANTITY_VELOCITY.type = QuantityType::Velocity;

    // additional averaging is performed in circumferential direction
    // for radial profiles (rotationally symmetric geometry)
    QUANTITY_VELOCITY_CIRCUMFERENTIAL.type = QuantityType::Velocity;
    QUANTITY_VELOCITY_CIRCUMFERENTIAL.average_circumferential = true;
    QUANTITY_VELOCITY_CIRCUMFERENTIAL.n_points_circumferential = N_POINTS_LINE_CIRCUMFERENTIAL;
    Tensor<1,dim,double> normal; normal[2] = 1.0;
    QUANTITY_VELOCITY_CIRCUMFERENTIAL.normal_vector = normal;

    axial_profile.quantities.push_back(&QUANTITY_VELOCITY);
    radial_profile_z1.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z2.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z3.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z4.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z5.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z6.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z7.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z8.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z9.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z10.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z11.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);
    radial_profile_z12.quantities.push_back(&QUANTITY_VELOCITY_CIRCUMFERENTIAL);

    // names
    axial_profile.name = "axial_profile";
    radial_profile_z1.name = "radial_profile_z1";
    radial_profile_z2.name = "radial_profile_z2";
    radial_profile_z3.name = "radial_profile_z3";
    radial_profile_z4.name = "radial_profile_z4";
    radial_profile_z5.name = "radial_profile_z5";
    radial_profile_z6.name = "radial_profile_z6";
    radial_profile_z7.name = "radial_profile_z7";
    radial_profile_z8.name = "radial_profile_z8";
    radial_profile_z9.name = "radial_profile_z9";
    radial_profile_z10.name = "radial_profile_z10";
    radial_profile_z11.name = "radial_profile_z11";
    radial_profile_z12.name = "radial_profile_z12";

    // insert lines
    line_plot_data.lines.push_back(axial_profile);
    line_plot_data.lines.push_back(radial_profile_z1);
    line_plot_data.lines.push_back(radial_profile_z2);
    line_plot_data.lines.push_back(radial_profile_z3);
    line_plot_data.lines.push_back(radial_profile_z4);
    line_plot_data.lines.push_back(radial_profile_z5);
    line_plot_data.lines.push_back(radial_profile_z6);
    line_plot_data.lines.push_back(radial_profile_z7);
    line_plot_data.lines.push_back(radial_profile_z8);
    line_plot_data.lines.push_back(radial_profile_z9);
    line_plot_data.lines.push_back(radial_profile_z10);
    line_plot_data.lines.push_back(radial_profile_z11);
    line_plot_data.lines.push_back(radial_profile_z12);
  }
}

// solve problem for DOMAIN 2 only (nozzle domain)
template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // call set_input_parameters() function for DOMAIN 2
  this->set_input_parameters(2);
}


/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity (const unsigned int  n_components = dim,
                           const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {
    srand(0); // initialize rand() to obtain reproducible results
  }

  virtual ~InitialSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double InitialSolutionVelocity<dim>::value(const Point<dim>   &p,
                                           const unsigned int component) const
{
  AssertThrow(dim==3, ExcMessage("Dimension has to be dim==3."));

  double result = 0.0;

  // flow in z-direction
  if(component == 2)
  {
    double radius = std::sqrt(p[0]*p[0]+p[1]*p[1]);

    // assume parabolic profile u(r) = u_max * [1-(r/R)^2]
    //  -> u_max = 2 * u_mean = 2 * flow_rate / area
    double const RADIUS = radius_function(p[2]);
    if(radius > RADIUS)
      radius = RADIUS;

    // parabolic velocity profile
    double const max_velocity_z = MAX_VELOCITY * std::pow(R_OUTER/RADIUS,2.0);
    result = max_velocity_z*(1.0-pow(radius/RADIUS,2.0));

    // Add perturbation (sine + random) for the precursor to initiate
    // a turbulent flow in case the Reynolds number is large enough
    // (otherwise, the perturbations will be damped and the flow becomes laminar).
    // According to first numerical results, the perturbed flow returns to a laminar
    // steady state in the precursor domain for Reynolds numbers Re_t = 500, 2000,
    // 3500, 5000, and 6500.
    if(p[2] <= Z2_PRECURSOR)
    {
      double const phi = std::atan2(p[1],p[0]);
      double const factor = 0.5;
      double perturbation = factor * max_velocity_z * std::sin(4.0*phi) * std::sin(8.0*numbers::PI*p[2]/LENGTH_PRECURSOR)
                            + factor * max_velocity_z * ((double)rand()/RAND_MAX-0.5)/0.5;

      // the perturbations should fulfill the Dirichlet boundary conditions
      perturbation *= (1.0-pow(radius/RADIUS,6.0));

      result += perturbation;
    }

  }

  return result;
}

#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"

template<int dim>
class InflowProfile : public Function<dim>
{
public:
  InflowProfile (const unsigned int n_components = dim,
                 const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {
    initialize_r_and_phi_values();
    initialize_velocity_values();
  }

  virtual ~InflowProfile(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const
  {
    // compute polar coordinates (r, phi) from point p
    // given in Cartesian coordinates (x, y) = inflow plane
    double const r = std::sqrt(p[0]*p[0] + p[1]*p[1]);
    double const phi = std::atan2(p[1],p[0]);

    double const result = linear_interpolation_2d_cylindrical(r,
                                                              phi,
                                                              R_VALUES,
                                                              PHI_VALUES,
                                                              VELOCITY_VALUES,
                                                              component);

    return result;
  }
};


/*
 *  Right-hand side function: Implements the body force vector occurring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations.
 *  Only relevant for precursor simulation.
 */
 template<int dim>
 class RightHandSide : public Function<dim>
 {
 public:
   RightHandSide (const double time = 0.)
     :
     Function<dim>(dim, time)
   {}

   virtual ~RightHandSide(){};

   virtual double value (const Point<dim>    & /*p*/,
                         const unsigned int  component = 0) const
   {
     double result = 0.0;

     // Channel flow with periodic bc in z-direction:
     // The flow is driven by body force in z-direction
     if(component==2)
     {
       result = FLOW_RATE_CONTROLLER.get_body_force();
     }

     return result;
   }
 };


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include "../../include/functionalities/one_sided_cylindrical_manifold.h"

/*
 *  Create grid for precursor domain (DOMAIN 1)
 */
template<int dim>
void create_grid_and_set_boundary_conditions_1(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /*
   *   PRECURSOR
   */
  Triangulation<2> tria_2d;
  GridGenerator::hyper_ball(tria_2d, Point<2>(), R_OUTER);
  GridGenerator::extrude_triangulation(tria_2d,N_CELLS_AXIAL_PRECURSOR+1,LENGTH_PRECURSOR,triangulation);
  Tensor<1,dim> offset = Tensor<1,dim>();
  offset[2] = Z1_PRECURSOR;
  GridTools::shift(offset,triangulation);

  /*
   *  MANIFOLDS
   */
  triangulation.set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin();cell != triangulation.end(); ++cell)
  {
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    {
      bool face_at_sphere_boundary = true;
      for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
      {
        Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);

        if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_OUTER) > 1e-12)
          face_at_sphere_boundary = false;
      }
      if (face_at_sphere_boundary)
      {
        face_ids.push_back(f);
        unsigned int manifold_id = manifold_ids.size() + 1;
        cell->set_all_manifold_ids(manifold_id);
        manifold_ids.push_back(manifold_id);
      }
    }
  }

  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i=0;i<manifold_ids.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],Point<dim>())));
        triangulation.set_manifold(manifold_ids[i],*(manifold_vec[i]));
      }
    }
  }

  /*
   *  BOUNDARY ID's
   */
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      // left boundary
      if ((std::fabs(cell->face(face_number)->center()[2] - Z1_PRECURSOR) < 1e-12))
      {
        cell->face(face_number)->set_boundary_id (0+10);
      }

      // right boundary
      if ((std::fabs(cell->face(face_number)->center()[2] - Z2_PRECURSOR) < 1e-12))
      {
        cell->face(face_number)->set_boundary_id (1+10);
      }
    }
  }

  GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 2, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  // perform global refinements
  triangulation.refine_global(n_refine_space);

  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  // fill boundary descriptor velocity
  // no slip boundaries at lower and upper wall with ID=0
  std::shared_ptr<Function<dim> > zero_function_velocity;
  zero_function_velocity.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_velocity));

  // fill boundary descriptor pressure
  // no slip boundaries at lower and upper wall with ID=0
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,pressure_bc_dudt));
}

/*
 *  Create grid for precursor domain (DOMAIN 2)
 */
template<int dim>
void create_grid_and_set_boundary_conditions_2(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /*
   *   Inflow
   */
  Triangulation<2> tria_2d_inflow;
  Triangulation<dim> tria_inflow;
  GridGenerator::hyper_ball(tria_2d_inflow, Point<2>(), R_OUTER);

  GridGenerator::extrude_triangulation(tria_2d_inflow,N_CELLS_AXIAL_INFLOW+1,LENGTH_INFLOW,tria_inflow);
  Tensor<1,dim> offset_inflow; offset_inflow[2] = Z1_INFLOW;
  GridTools::shift(offset_inflow,tria_inflow);

  Triangulation<dim> * current_tria = &tria_inflow;

  /*
   *   Cone
   */
  Triangulation<2> tria_2d_cone;
  Triangulation<dim> tria_cone;
  GridGenerator::hyper_ball(tria_2d_cone, Point<2>(), R_OUTER);

  GridGenerator::extrude_triangulation(tria_2d_cone,N_CELLS_AXIAL_CONE+1,LENGTH_CONE,tria_cone);
  Tensor<1,dim> offset_cone; offset_cone[2] = Z1_CONE;
  GridTools::shift(offset_cone,tria_cone);

  // apply conical geometry: stretch vertex positions according to z-coordinate
  for (typename Triangulation<dim>::cell_iterator cell = tria_cone.begin(); cell != tria_cone.end(); ++cell)
  {
    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      if(cell->vertex(v)[2] > Z1_CONE+1.e-10)
      {
        Point<dim> point_2d;
        double const z = cell->vertex(v)[2];
        point_2d[2] = z;

        if(std::abs((cell->vertex(v) - point_2d).norm() - 2.485281374239e-03) < 1.e-10 ||
           std::abs((cell->vertex(v) - point_2d).norm() - R_OUTER) < 1.e-10)
        {
          cell->vertex(v)[0] *= 1.0 - (cell->vertex(v)[2]-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER;
          cell->vertex(v)[1] *= 1.0 - (cell->vertex(v)[2]-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER;
        }
      }
    }
  }

  /*
   *   Throat
   */
  Triangulation<2> tria_2d_throat;
  Triangulation<dim> tria_throat;
  GridGenerator::hyper_ball(tria_2d_throat, Point<2>(), R_INNER);

  GridGenerator::extrude_triangulation(tria_2d_throat,N_CELLS_AXIAL_THROAT+1,LENGTH_THROAT,tria_throat);
  Tensor<1,dim> offset_throat; offset_throat[2] = Z1_THROAT;
  GridTools::shift(offset_throat,tria_throat);

  /*
   *   OUTFLOW
   */
  const unsigned int n_cells_circle = 4;
  double const R_1 = R_INNER + 1.0/3.0*(R_OUTER-R_INNER);
  double const R_2 = R_INNER + 2.0/3.0*(R_OUTER-R_INNER);

  Triangulation<2> tria_2d_outflow_inner, circle_1, circle_2, circle_3, tria_tmp_2d_1, tria_tmp_2d_2, tria_2d_outflow;
  GridGenerator::hyper_ball(tria_2d_outflow_inner, Point<2>(), R_INNER);

  GridGenerator::hyper_shell(circle_1, Point<2>(), R_INNER, R_1, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_1);
  GridGenerator::hyper_shell(circle_2, Point<2>(), R_1, R_2, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_2);
  GridGenerator::hyper_shell(circle_3, Point<2>(), R_2, R_OUTER, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_3);

  // merge 2d triangulations
  GridGenerator::merge_triangulations (tria_2d_outflow_inner, circle_1, tria_tmp_2d_1);
  GridGenerator::merge_triangulations (circle_2, circle_3, tria_tmp_2d_2);
  GridGenerator::merge_triangulations (tria_tmp_2d_1, tria_tmp_2d_2, tria_2d_outflow);

  // extrude in z-direction
  Triangulation<dim> tria_outflow;
  GridGenerator::extrude_triangulation(tria_2d_outflow,N_CELLS_AXIAL_OUTFLOW+1,LENGTH_OUTFLOW,tria_outflow);
  Tensor<1,dim> offset_outflow; offset_outflow[2] = Z1_OUTFLOW;
  GridTools::shift(offset_outflow,tria_outflow);

  /*
   *  MERGE TRIANGULATIONS
   */
  Triangulation<dim> tria_tmp, tria_tmp2;
  GridGenerator::merge_triangulations (tria_inflow, tria_cone, tria_tmp);
  GridGenerator::merge_triangulations (tria_tmp, tria_throat, tria_tmp2);
  GridGenerator::merge_triangulations (tria_tmp2, tria_outflow, triangulation);

  /*
   *  MANIFOLDS
   */
  current_tria = &triangulation;
  current_tria->set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids_cone;
  std::vector<unsigned int> face_ids_cone;
  std::vector<double> radius_0_cone;
  std::vector<double> radius_1_cone;

  for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin();cell != current_tria->end(); ++cell)
  {
    // INFLOW
    if(cell->center()[2] < Z2_INFLOW)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_OUTER) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    // CONE
    else if(cell->center()[2] > Z1_CONE && cell->center()[2] < Z2_CONE)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        double min_z = std::numeric_limits<double>::max();
        double max_z = - std::numeric_limits<double>::max();

        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          double const z = cell->face(f)->vertex(v)[2];
          if(z > max_z)
            max_z = z;
          if(z < min_z)
            min_z = z;

          Point<dim> point = Point<dim>(0,0,z);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-radius_function(z)) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids_cone.push_back(f);
          unsigned int manifold_id = MANIFOLD_ID_OFFSET_CONE + manifold_ids_cone.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids_cone.push_back(manifold_id);
          radius_0_cone.push_back(radius_function(min_z));
          radius_1_cone.push_back(radius_function(max_z));
        }
      }
    }
    // THROAT
    else if(cell->center()[2] > Z1_THROAT && cell->center()[2] < Z2_THROAT)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_INNER) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    // OUTFLOW
    else if(cell->center()[2] > Z1_OUTFLOW && cell->center()[2] < Z2_OUTFLOW)
    {
      Point<dim> point2 = Point<dim>(0,0,cell->center()[2]);

      // cylindrical manifold for outer cell layers
      if((cell->center()-point2).norm() > R_INNER/std::sqrt(2.0))
        cell->set_all_manifold_ids(MANIFOLD_ID_CYLINDER);

      // one-sided cylindrical manifold for core region
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_INNER) > 1e-12 ||
              (cell->center()-point2).norm() > R_INNER/std::sqrt(2.0))
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Should not arrive here."));
    }
  }

  // one-sided spherical manifold
  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i=0;i<manifold_ids.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin(); cell != current_tria->end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        Point<dim> center = Point<dim>();
        manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],center)));
        current_tria->set_manifold(manifold_ids[i],*(manifold_vec[i]));
      }
    }
  }

  // conical manifold
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec_cone;
  manifold_vec_cone.resize(manifold_ids_cone.size());

  for(unsigned int i=0;i<manifold_ids_cone.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin(); cell != current_tria->end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids_cone[i])
      {
        Point<dim> center = Point<dim>();
        manifold_vec_cone[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedConicalManifold<dim>(cell,face_ids_cone[i],center,radius_0_cone[i],radius_1_cone[i])));
        current_tria->set_manifold(manifold_ids_cone[i],*(manifold_vec_cone[i]));
      }
    }
  }

  // set cylindrical manifold
  static std::shared_ptr<Manifold<dim> > cylinder_manifold;
  cylinder_manifold = std::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new MyCylindricalManifold<dim>(Point<dim>())));
  current_tria->set_manifold(MANIFOLD_ID_CYLINDER, *cylinder_manifold);



  /*
   *  BOUNDARY ID's
   */
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      // inflow boundary on the left has ID = 1
      if ((std::fabs(cell->face(f)->center()[2] - Z1_INFLOW)< 1e-12))
      {
        cell->face(f)->set_boundary_id (1);
      }

      // outflow boundary on the right has ID = 2
      if ((std::fabs(cell->face(f)->center()[2] - Z2_OUTFLOW)< 1e-12))
      {
        cell->face(f)->set_boundary_id (2);
      }
    }
  }

  // perform global refinements
  triangulation.refine_global(n_refine_space);

  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  // fill boundary descriptor velocity
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > zero_function_velocity;
  zero_function_velocity.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_velocity));

  // inflow boundary condition at left boundary with ID=1: prescribe velocity profile which
  // is obtained as the results of the simulation on DOMAIN 1
  std::shared_ptr<Function<dim> > inflow_profile;
  inflow_profile.reset(new InflowProfile<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,inflow_profile));

  // outflow boundary condition at right boundary with ID=2
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_velocity));

  // fill boundary descriptor pressure
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,pressure_bc_dudt));

  // inflow boundary condition at left boundary with ID=1
  // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
  // we assume that this is negligible when using the dual splitting scheme
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,pressure_bc_dudt));

  // outflow boundary condition at right boundary with ID=2: set pressure to zero
  std::shared_ptr<Function<dim> > zero_function_pressure;
  zero_function_pressure.reset(new ZeroFunction<dim>(1));
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_pressure));
}

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  // call respective function for DOMAIN 2
  create_grid_and_set_boundary_conditions_2(triangulation,
                                            n_refine_space,
                                            boundary_descriptor_velocity,
                                            boundary_descriptor_pressure,
                                            periodic_faces);
}

template<int dim>
void set_field_functions_1(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  // prescribe body force for the turbulent channel (DOMAIN 1) to
  // adjust the desired flow rate
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_field_functions_2(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  // no body forces for the second domain
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new ZeroFunction<dim>(dim));

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // call respective function for DOMAIN 2
  set_field_functions_2(field_functions);
}


template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new ZeroFunction<dim>(1));
}

// Postprocessor

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"
#include "../../include/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h"

template<int dim>
struct PostProcessorDataFDA
{
  PostProcessorData<dim> pp_data;
  InflowData<dim> inflow_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotData<dim> line_plot_data;
};

template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
class PostProcessorFDA : public PostProcessor<dim, fe_degree_u, fe_degree_p, Number>
{
public:
  PostProcessorFDA(PostProcessorDataFDA<dim> const & pp_data_in)
    :
    PostProcessor<dim,fe_degree_u,fe_degree_p, Number>(pp_data_in.pp_data),
    pp_data_fda(pp_data_in),
    time_old(START_TIME_PRECURSOR)
  {
    inflow_data_calculator.reset(new InflowDataCalculator<dim,Number>(pp_data_in.inflow_data));
  }

  void setup(DoFHandler<dim> const                     &dof_handler_velocity_in,
             DoFHandler<dim> const                     &dof_handler_pressure_in,
             Mapping<dim> const                        &mapping_in,
             MatrixFree<dim,Number> const              &matrix_free_data_in,
             DofQuadIndexData const                    &dof_quad_index_data_in,
             std::shared_ptr<AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    PostProcessor<dim,fe_degree_u,fe_degree_p,Number>::setup(
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // inflow data
    inflow_data_calculator->setup(dof_handler_velocity_in,mapping_in);

    // calculation of mean velocity
    mean_velocity_calculator.reset(new MeanVelocityCalculator<dim,fe_degree_u,Number>(
        matrix_free_data_in, dof_quad_index_data_in, pp_data_fda.mean_velocity_data));

    // evaluation of results along lines
    line_plot_calculator_statistics.reset(new LinePlotCalculatorStatistics<dim>(
        dof_handler_velocity_in, dof_handler_pressure_in, mapping_in));
    line_plot_calculator_statistics->setup(pp_data_fda.line_plot_data);
  }

  void do_postprocessing(parallel::distributed::Vector<Number> const   &velocity,
                         parallel::distributed::Vector<Number> const   &intermediate_velocity,
                         parallel::distributed::Vector<Number> const   &pressure,
                         parallel::distributed::Vector<Number> const   &vorticity,
                         std::vector<SolutionField<dim,Number> > const &additional_fields,
                         double const                                  time,
                         int const                                     time_step_number)
  {
    PostProcessor<dim,fe_degree_u,fe_degree_p,Number>::do_postprocessing(
	      velocity,
        intermediate_velocity,
        pressure,
        vorticity,
        additional_fields,
        time,
        time_step_number);

    if(USE_PRECURSOR_SIMULATION == true)
    {
      // inflow data
      inflow_data_calculator->calculate(velocity);
    }
    else
    {
      if(USE_RANDOM_PERTURBATION==true)
        initialize_velocity_values();
    }

    if(pp_data_fda.mean_velocity_data.calculate == true)
    {
      // calculation of flow rate
      FLOW_RATE = AREA_INFLOW*mean_velocity_calculator->calculate_mean_velocity_volume(velocity,time);

      // set time step size for flow rate controller
      TIME_STEP_FLOW_RATE_CONTROLLER = time-time_old;
      time_old = time;

      // update body force
      FLOW_RATE_CONTROLLER.update_body_force();
    }

    // evaluation of results along lines
    line_plot_calculator_statistics->evaluate(velocity,pressure,time,time_step_number);
  }

private:
  // postprocessor data supplemented with data required for FDA benchmark
  PostProcessorDataFDA<dim> pp_data_fda;
  // interpolate velocity field to a predefined set of interpolation points
  std::shared_ptr<InflowDataCalculator<dim, Number> > inflow_data_calculator;
  // calculate flow rate in precursor domain so that the flow rate can be
  // dynamically adjusted by a flow rate controller.
  std::shared_ptr<MeanVelocityCalculator<dim,fe_degree_u,Number> > mean_velocity_calculator;
  // the low rate controller needs the time step size, so we have to store the previous time instant
  double time_old;
  // evaluation of results along lines
  std::shared_ptr<LinePlotCalculatorStatistics<dim> > line_plot_calculator_statistics;
};

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim,Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  // basic modules
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  // FDA specific modules
  PostProcessorDataFDA<dim> pp_data_fda;
  pp_data_fda.pp_data = pp_data;
  pp_data_fda.inflow_data = param.inflow_data;
  pp_data_fda.mean_velocity_data = param.mean_velocity_data;
  pp_data_fda.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessorFDA<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,Number>(pp_data_fda));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
