/*
 * backward_facing_step_two_domains.h
 *
 *  Created on: Oct 14, 2016
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

// set the polynomial degree of the shape functions for velocity and pressure.
// currently, one has to use the same polynomial degree for both domains.
unsigned int const FE_DEGREE_VELOCITY = 2;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// DOMAIN 1: turbulent channel problem: used to generate inflow data for the BFS
// DOMAIN 2: backward facing step (using results of the turbulent channel flow as velocity inflow profile)

// set the number of refine levels for DOMAIN 1
unsigned int const REFINE_STEPS_SPACE_DOMAIN1 = 3;

// set the number of refine levels for DOMAIN 2
unsigned int const REFINE_STEPS_SPACE_DOMAIN2 = 3;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
double const PI = numbers::PI;

// Height H
double const H = 0.041;

// channel
double const LENGTH_CHANNEL = 2.0*PI*H;
double const HEIGHT_CHANNEL = 2.0*H;
double const WIDTH_CHANNEL = 4.0*H;

// use a gap between both geometries for visualization purposes
double const GAP_CHANNEL_BFS = 2.0*H;

// backward facing step geometry
double const LENGTH_BFS_DOWN = 20.0*H;
double const LENGTH_BFS_UP = 2.0*H;
double const HEIGHT_BFS_STEP = H;
double const HEIGHT_BFS_INFLOW = HEIGHT_CHANNEL;
double const WIDTH_BFS = WIDTH_CHANNEL;

double const X1_COORDINATE_INFLOW = - LENGTH_BFS_UP;
double const X1_COORDINATE_OUTFLOW = LENGTH_BFS_DOWN;
double const X1_COORDINATE_OUTFLOW_CHANNEL = - LENGTH_BFS_UP - GAP_CHANNEL_BFS;

// mesh stretching parameters
bool use_grid_stretching_in_y_direction = true;

double const GAMMA_LOWER = 60.0;
double const GAMMA_UPPER = 40.0;

// consider a friction Reynolds number of Re_tau = u_tau * H / nu = 290
// and body force f = tau_w/H with tau_w = u_tau^2.
double const VISCOSITY = 1.5268e-5;

// estimate the maximum velocity
double const MAX_VELOCITY = 2.0;

// times
double const START_TIME = 0.0;
double const SAMPLE_START_TIME = 2.0;
double const END_TIME = 6.0;
unsigned int const SAMPLE_EVERY_TIMESTEPS = 10;

// postprocessing and output
QuantityStatistics<DIMENSION> QUANTITY_VELOCITY;
QuantityStatisticsSkinFriction<3> QUANTITY_SKIN_FRICTION;
QuantityStatistics<DIMENSION> QUANTITY_REYNOLDS;
QuantityStatistics<DIMENSION> QUANTITY_PRESSURE;
QuantityStatisticsPressureCoefficient<DIMENSION> QUANTITY_PRESSURE_COEFF;
const unsigned int N_POINTS_LINE = 101;

// output folders and names
std::string OUTPUT_FOLDER = "output/bfs/test2/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_1 = "precursor";
std::string OUTPUT_NAME_2 = "bfs";
bool WRITE_VTU_OUTPUT = false;
unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 100; //1e3;

// data structures that we need in order to apply the velocity
// inflow profile (we currently use global variables for this purpose)
unsigned int N_POINTS_Y = 101;
unsigned int N_POINTS_Z = N_POINTS_Y;
std::vector<double> Y_VALUES(N_POINTS_Y);
std::vector<double> Z_VALUES(N_POINTS_Z);
std::vector<Tensor<1,DIMENSION,double> > VELOCITY_VALUES(N_POINTS_Y*N_POINTS_Z);

// initial vectors
void initialize_y_and_z_values()
{
  AssertThrow(N_POINTS_Y >= 2, ExcMessage("Variable N_POINTS_Y is invalid"));
  AssertThrow(N_POINTS_Z >= 2, ExcMessage("Variable N_POINTS_Z is invalid"));

  for(unsigned int i=0; i<N_POINTS_Y; ++i)
    Y_VALUES[i] = double(i)/double(N_POINTS_Y-1)*HEIGHT_CHANNEL;

  for(unsigned int i=0; i<N_POINTS_Z; ++i)
    Z_VALUES[i] = -WIDTH_CHANNEL/2.0 + double(i)/double(N_POINTS_Z-1)*WIDTH_CHANNEL;
}

void initialize_velocity_values()
{
  AssertThrow(N_POINTS_Y >= 2, ExcMessage("Variable N_POINTS_Y is invalid"));
  AssertThrow(N_POINTS_Z >= 2, ExcMessage("Variable N_POINTS_Z is invalid"));

  for(unsigned int iy=0; iy<N_POINTS_Y; ++iy)
  {
    for(unsigned int iz=0; iz<N_POINTS_Z; ++iz)
    {
      Tensor<1,DIMENSION,double> velocity;
      VELOCITY_VALUES[iy*N_POINTS_Y + iz] = velocity;
    }
  }
}

// we do not need this function here (but have to implement it)
template<int dim>
void InputParameters<dim>::set_input_parameters()
{

}

/*
 *  To set input parameters for DOMAIN 1 and DOMAIN 2, use
 *
 *  if(domain_id == 1){}
 *  else if(domain_id == 2){}
 *
 *  Most of the input parameters are the same for both domains!
 */
template<int dim>
void InputParameters<dim>::set_input_parameters(unsigned int const domain_id)
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  use_outflow_bc_convective_term = true;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; // BDFDualSplittingScheme; //BDFPressureCorrection; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = MAX_VELOCITY;
  cfl = 0.5;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // variant Direct allows to use larger time step
  // sizes due to CFL condition at inflow boundary
  type_dirichlet_bc_convective = TypeDirichletBCs::Mirror;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  if(domain_id == 1)
    pure_dirichlet_bc = true;
  else if(domain_id == 2)
    pure_dirichlet_bc = false;

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
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix; //BlockJacobi; //PointJacobi; //InverseMassMatrix;
  update_preconditioner_projection = true;
  solver_data_projection = SolverData(1e4, 1.e-12, 1.e-6, 100);


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //GeometricMultigrid;
  solver_data_viscous = SolverData(1e4, 1.e-12, 1.e-6, 100);


  // PRESSURE-CORRECTION SCHEME

  // formulation
  order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
  rotational_formulation = true;

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
  update_preconditioner_momentum = false;

  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::GMRES; //GMRES; //FGMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // OUTPUT AND POSTPROCESSING
  if(domain_id == 1)
  {
    // write output for visualization of results
    output_data.write_output = WRITE_VTU_OUTPUT;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_1;
    output_data.output_start_time = START_TIME;
    output_data.output_interval_time = (END_TIME-START_TIME)/60;
    output_data.write_divergence = true;
    output_data.write_q_criterion = true;
    output_data.degree = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS;

    // turbulent channel statistics
    turb_ch_data.calculate_statistics = true;
    turb_ch_data.cells_are_stretched = use_grid_stretching_in_y_direction;
    turb_ch_data.sample_start_time = SAMPLE_START_TIME;
    turb_ch_data.sample_end_time = END_TIME;
    turb_ch_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
    turb_ch_data.viscosity = VISCOSITY;
    turb_ch_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME_1;

    // use turbulent channel data to prescribe inflow velocity for BFS
    inflow_data.write_inflow_data = true;
    inflow_data.normal_direction = 0; /* x-direction */
    inflow_data.normal_coordinate = X1_COORDINATE_OUTFLOW_CHANNEL;
    inflow_data.n_points_y = N_POINTS_Y;
    inflow_data.n_points_z = N_POINTS_Z;
    inflow_data.y_values = &Y_VALUES;
    inflow_data.z_values = &Z_VALUES;
    inflow_data.array = &VELOCITY_VALUES;
  }
  else if(domain_id == 2)
  {
    // write output for visualization of results
    output_data.write_output = WRITE_VTU_OUTPUT;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_2;
    output_data.output_start_time = START_TIME;
    output_data.output_interval_time = (END_TIME-START_TIME)/60;
    output_data.write_divergence = true;
    output_data.write_q_criterion = true;
    output_data.degree = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS;

    // line plot data: calculate statistics along lines
    line_plot_data.write_output = true;
    line_plot_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME_2;
    line_plot_data.statistics_data.calculate_statistics = true;
    line_plot_data.statistics_data.sample_start_time = SAMPLE_START_TIME;
    line_plot_data.statistics_data.sample_end_time = END_TIME;
    line_plot_data.statistics_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
    line_plot_data.statistics_data.write_output_every_timesteps = SAMPLE_EVERY_TIMESTEPS*10;

    // mean velocity
    QUANTITY_VELOCITY.type = QuantityType::Velocity;
    QUANTITY_VELOCITY.averaging_direction = 2;

    // reynolds stresses
    QUANTITY_REYNOLDS.type = QuantityType::ReynoldsStresses;
    QUANTITY_REYNOLDS.averaging_direction = 2;
    
    // skin friction
    Tensor<1,dim,double> normal; normal[1] = 1.0;
    Tensor<1,dim,double> tangent; tangent[0] = 1.0;
    QUANTITY_SKIN_FRICTION.type = QuantityType::SkinFriction;
    QUANTITY_SKIN_FRICTION.averaging_direction = 2;
    QUANTITY_SKIN_FRICTION.normal_vector = normal;
    QUANTITY_SKIN_FRICTION.tangent_vector = tangent;
    QUANTITY_SKIN_FRICTION.viscosity = VISCOSITY;

    // mean pressure
    QUANTITY_PRESSURE.type = QuantityType::Pressure;
    QUANTITY_PRESSURE.averaging_direction = 2;

    // mean pressure coefficient
    QUANTITY_PRESSURE_COEFF.type = QuantityType::PressureCoefficient;
    QUANTITY_PRESSURE_COEFF.averaging_direction = 2;
    QUANTITY_PRESSURE_COEFF.reference_point = Point<DIMENSION>(X1_COORDINATE_INFLOW,0,0);

    // lines
    Line<dim> vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, vel_7, vel_8, vel_9, vel_10, vel_11, Cp_1, Cp_2, Cf;

    // begin and end points of all lines
    vel_0.begin = Point<dim> (X1_COORDINATE_INFLOW,   0,0);
    vel_0.end =   Point<dim> (X1_COORDINATE_INFLOW, 2*H,0);
    vel_1.begin = Point<dim> (0*H,   0,0);
    vel_1.end =   Point<dim> (0*H, 2*H,0);
    vel_2.begin = Point<dim> (1*H,-1*H,0);
    vel_2.end =   Point<dim> (1*H, 2*H,0);
    vel_3.begin = Point<dim> (2*H,-1*H,0);
    vel_3.end =   Point<dim> (2*H, 2*H,0);
    vel_4.begin = Point<dim> (3*H,-1*H,0);
    vel_4.end =   Point<dim> (3*H, 2*H,0);
    vel_5.begin = Point<dim> (4*H,-1*H,0);
    vel_5.end =   Point<dim> (4*H, 2*H,0);
    vel_6.begin = Point<dim> (5*H,-1*H,0);
    vel_6.end =   Point<dim> (5*H, 2*H,0);
    vel_7.begin = Point<dim> (6*H,-1*H,0);
    vel_7.end =   Point<dim> (6*H, 2*H,0);
    vel_8.begin = Point<dim> (7*H,-1*H,0);
    vel_8.end =   Point<dim> (7*H, 2*H,0);
    vel_9.begin = Point<dim> (8*H,-1*H,0);
    vel_9.end =   Point<dim> (8*H, 2*H,0);
    vel_10.begin = Point<dim> (9*H,-1*H,0);
    vel_10.end =   Point<dim> (9*H, 2*H,0);
    vel_11.begin = Point<dim> (10*H,-1*H,0);
    vel_11.end =   Point<dim> (10*H, 2*H,0);
    Cp_1.begin = Point<dim> (X1_COORDINATE_INFLOW,0,0);
    Cp_1.end =   Point<dim> (0,0,0);
    Cp_2.begin = Point<dim> (0,-H,0);
    Cp_2.end =   Point<dim> (X1_COORDINATE_OUTFLOW,-H,0);
    Cf.begin = Point<dim> (0,-H,0);
    Cf.end =   Point<dim> (X1_COORDINATE_OUTFLOW,-H,0);

    // set the number of points along the lines
    vel_0.n_points = N_POINTS_LINE;
    vel_1.n_points = N_POINTS_LINE;
    vel_2.n_points = N_POINTS_LINE;
    vel_3.n_points = N_POINTS_LINE;
    vel_4.n_points = N_POINTS_LINE;
    vel_5.n_points = N_POINTS_LINE;
    vel_6.n_points = N_POINTS_LINE;
    vel_7.n_points = N_POINTS_LINE;
    vel_8.n_points = N_POINTS_LINE;
    vel_9.n_points = N_POINTS_LINE;
    vel_10.n_points = N_POINTS_LINE;
    vel_11.n_points = N_POINTS_LINE;
    Cp_1.n_points = N_POINTS_LINE;
    Cp_2.n_points = N_POINTS_LINE;
    Cf.n_points = N_POINTS_LINE;

    // set the quantities that we want to compute along the lines
    vel_0.quantities.push_back(&QUANTITY_VELOCITY);
    vel_0.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_1.quantities.push_back(&QUANTITY_VELOCITY);
    vel_1.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_2.quantities.push_back(&QUANTITY_VELOCITY);
    vel_2.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_3.quantities.push_back(&QUANTITY_VELOCITY);
    vel_3.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_4.quantities.push_back(&QUANTITY_VELOCITY);
    vel_4.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_5.quantities.push_back(&QUANTITY_VELOCITY);
    vel_5.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_6.quantities.push_back(&QUANTITY_VELOCITY);
    vel_6.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_7.quantities.push_back(&QUANTITY_VELOCITY);
    vel_7.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_8.quantities.push_back(&QUANTITY_VELOCITY);
    vel_8.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_9.quantities.push_back(&QUANTITY_VELOCITY);
    vel_9.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_10.quantities.push_back(&QUANTITY_VELOCITY);
    vel_10.quantities.push_back(&QUANTITY_REYNOLDS);
    vel_11.quantities.push_back(&QUANTITY_VELOCITY);
    vel_11.quantities.push_back(&QUANTITY_REYNOLDS);
    Cp_1.quantities.push_back(&QUANTITY_PRESSURE);
    Cp_1.quantities.push_back(&QUANTITY_PRESSURE_COEFF);
    Cp_2.quantities.push_back(&QUANTITY_PRESSURE);
    Cp_2.quantities.push_back(&QUANTITY_PRESSURE_COEFF);
    Cf.quantities.push_back(&QUANTITY_SKIN_FRICTION);

    // set line names
    vel_0.name = "vel_0";
    vel_1.name = "vel_1";
    vel_2.name = "vel_2";
    vel_3.name = "vel_3";
    vel_4.name = "vel_4";
    vel_5.name = "vel_5";
    vel_6.name = "vel_6";
    vel_7.name = "vel_7";
    vel_8.name = "vel_8";
    vel_9.name = "vel_9";
    vel_10.name = "vel_10";
    vel_11.name = "vel_11";
    Cp_1.name = "Cp_1";
    Cp_2.name = "Cp_2";
    Cf.name = "Cf";

    // insert lines
    line_plot_data.lines.push_back(vel_0);
    line_plot_data.lines.push_back(vel_1);
    line_plot_data.lines.push_back(vel_2);
    line_plot_data.lines.push_back(vel_3);
    line_plot_data.lines.push_back(vel_4);
    line_plot_data.lines.push_back(vel_5);
    line_plot_data.lines.push_back(vel_6);
    line_plot_data.lines.push_back(vel_7);
    line_plot_data.lines.push_back(vel_8);
    line_plot_data.lines.push_back(vel_9);
    line_plot_data.lines.push_back(vel_10);
    line_plot_data.lines.push_back(vel_11);
    line_plot_data.lines.push_back(Cp_1);
    line_plot_data.lines.push_back(Cp_2);
    line_plot_data.lines.push_back(Cf);
  }
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
  }

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = 0.0;
    double const y = (p[1] - HEIGHT_CHANNEL/2.0)/(HEIGHT_CHANNEL/2.0);

    if(dim==3)
    {
      if(component == 0)
      {
        double factor = 0.5;

        if(std::abs(y)<1.0)
          result = -MAX_VELOCITY*(pow(y,6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-0.5)*factor);
        else
          result = 0.0;
      }

  //    double factor = 0.5;
  //    double const x = p[0]/LENGTH_CHANNEL;
  //    double const z = p[2]/WIDTH_CHANNEL;
  //
  //    if(component == 0)
  //    {
  //      if(std::abs(y)<1.0)
  //        result = -MAX_VELOCITY*(pow(y,6.0)-1.0)*(1.0 + (((double)rand()/RAND_MAX-1.0) + std::sin(z*8.)*0.5)*factor);
  //    }
  //    if(component == 2)
  //    {
  //      if(std::abs(y)<1.0)
  //        result = -MAX_VELOCITY*(pow(y,6.0)-1.0)*std::sin(x*8.)*0.5*factor;
  //    }
    }
    else
    {
      AssertThrow(false, ExcMessage("Dimension has to be dim==3."));
    }

    return result;
  }
};


#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"

template<int dim>
class InflowProfile : public Function<dim>
{
public:
  InflowProfile (const unsigned int  n_components = dim,
                 const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {
    initialize_y_and_z_values();
    initialize_velocity_values();
  }

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = linear_interpolation_2d_cartesian(p,Y_VALUES,Z_VALUES,VELOCITY_VALUES,component);

    return result;
  }
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  component = 0) const
  {
    double result = 0.0;

    //channel flow with periodic bc
    if(component==0)
      return 0.2844518;
    else
      return 0.0;

    return result;
  }
};

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

/*
 *  maps eta in [-H, 2*H] --> y in [-H,2*H]
 */
double grid_transform_y(const double &eta)
{
  double y = 0.0;
  double gamma, xi;
  if (eta < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi = -0.5*H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi = H;
  }
  y = xi * (1.0 - (std::tanh(gamma*(xi-eta))/std::tanh(gamma*xi)));
  return y;
}

/*
 *  grid transform function for turbulent channel statistics
 *  requires that the input parameter is 0 < xi < 1
 */
double grid_transform_turb_channel(const double &xi)
{
  // map xi in [0,1] --> eta in [0, 2H]
  double eta = HEIGHT_CHANNEL * xi;
  return grid_transform_y(eta);
}

/*
 * inverse mapping:
 *
 *  maps y in [-H,2*H] --> eta in [-H,2*H]
 */
double inverse_grid_transform_y(const double &y)
{
  double eta = 0.0;
  double gamma, xi;
  if (y < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi = -0.5*H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi = H;
  }
  eta = xi - (1.0/gamma)*std::atanh((1.0-y/xi)*std::tanh(gamma*xi));
  return eta;
}

#include <deal.II/grid/manifold_lib.h>

template <int dim>
class ManifoldTurbulentChannel : public ChartManifold<dim,dim,dim>
{
public:
  ManifoldTurbulentChannel()
  { }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim> push_forward(const Point<dim> &xi) const
  {
    Point<dim> x = xi;
    x[1] = grid_transform_y(xi[1]);

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d
   */
  Point<dim> pull_back(const Point<dim> &x) const
  {
    Point<dim> xi = x;
    xi[1] = inverse_grid_transform_y(x[1]);

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std_cxx14::make_unique<ManifoldTurbulentChannel<dim>>();
  }
};

template<int dim>
void create_grid_and_set_boundary_conditions_1(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /* --------------- Generate grid ------------------- */
  AssertThrow(dim==3, ExcMessage("NotImplemented"));

  Tensor<1,dim> dimensions;
  dimensions[0] = LENGTH_CHANNEL;
  dimensions[1] = HEIGHT_CHANNEL;
  dimensions[2] = WIDTH_CHANNEL;

  Tensor<1,dim> center;
  center[0] = - (LENGTH_BFS_UP + GAP_CHANNEL_BFS + LENGTH_CHANNEL/2.0);
  center[1] = HEIGHT_CHANNEL/2.0;

  GridGenerator::subdivided_hyper_rectangle (*triangulation,
                                             std::vector<unsigned int>({2,1,1}), //refinements
                                             Point<dim>(center-dimensions/2.0),
                                             Point<dim>(center+dimensions/2.0));

  if(use_grid_stretching_in_y_direction == true)
  {
    // manifold
    unsigned int manifold_id = 1;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold;
    triangulation->set_manifold(manifold_id, manifold);
  }

  // set boundary ID's: periodicity
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      // periodicity in x-direction (add 10 to avoid conflicts with other boundaries)
      if(std::fabs(cell->face(face_number)->center()(0) - (center[0] - dimensions[0]/2.0)) < 1.e-12)
        cell->face(face_number)->set_all_boundary_ids (0+10);
      // periodicity in x-direction (add 10 to avoid conflicts with other boundaries)
      if(std::fabs(cell->face(face_number)->center()(0) - (center[0] + dimensions[0]/2.0)) < 1.e-12)
        cell->face(face_number)->set_all_boundary_ids (1+10);

      // periodicity in z-direction (add 10 to avoid conflicts with other boundaries)
      if(std::fabs(cell->face(face_number)->center()(2) - (center[2] - dimensions[2]/2.0)) < 1.e-12)
        cell->face(face_number)->set_all_boundary_ids (2+10);
      // periodicity in z-direction (add 10 to avoid conflicts with other boundaries)
      if(std::fabs(cell->face(face_number)->center()(2) - (center[2] + dimensions[2]/2.0)) < 1.e-12)
        cell->face(face_number)->set_all_boundary_ids (3+10);
    }
  }

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 0, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2+10, 3+10, 2, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  // perform global refinements: use one level finer for the channel
  triangulation->refine_global(n_refine_space);

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  // no slip boundaries at lower and upper wall with ID=0
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  // no slip boundaries at lower and upper wall with ID=0
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
}

template<int dim>
void create_grid_and_set_boundary_conditions_2(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /* --------------- Generate grid ------------------- */
  if(dim==2)
  {
    AssertThrow(false, ExcMessage("NotImplemented"));
  }
  else if(dim==3)
  {
    Triangulation<dim> tria_1, tria_2, tria_3;

    // inflow part of BFS
    GridGenerator::subdivided_hyper_rectangle(tria_1,
                                              std::vector<unsigned int>({1,1,1}),
                                              Point<dim>(-LENGTH_BFS_UP,0.0,-WIDTH_BFS/2.0),
                                              Point<dim>(0.0,HEIGHT_BFS_INFLOW,WIDTH_BFS/2.0));

    // downstream part of BFS (upper)
    GridGenerator::subdivided_hyper_rectangle(tria_2,
                                              std::vector<unsigned int>({10,1,1}),
                                              Point<dim>(0.0,0.0,-WIDTH_BFS/2.0),
                                              Point<dim>(LENGTH_BFS_DOWN,HEIGHT_BFS_INFLOW,WIDTH_BFS/2.0));

    // downstream part of BFS (lower = step)
    GridGenerator::subdivided_hyper_rectangle(tria_3,
                                              std::vector<unsigned int>({10,1,1}),
                                              Point<dim>(0.0,0.0,-WIDTH_BFS/2.0),
                                              Point<dim>(LENGTH_BFS_DOWN,-HEIGHT_BFS_STEP,WIDTH_BFS/2.0));

    Triangulation<dim> tmp1;
    GridGenerator::merge_triangulations (tria_1, tria_2, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_3, *triangulation);
  }

  // set boundary ID's
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      // outflow boundary on the right has ID = 1
      if ((std::fabs(cell->face(face_number)->center()(0) - X1_COORDINATE_OUTFLOW)< 1.e-12))
        cell->face(face_number)->set_boundary_id (1);
      // inflow boundary on the left has ID = 2
      if ((std::fabs(cell->face(face_number)->center()(0) - X1_COORDINATE_INFLOW)< 1.e-12))
        cell->face(face_number)->set_boundary_id (2);

      // periodicity in z-direction (add 10 to avoid conflicts with other boundaries)
      if((std::fabs(cell->face(face_number)->center()(2) - WIDTH_BFS/2.0)< 1.e-12))
        cell->face(face_number)->set_all_boundary_ids (2+10);
      // periodicity in z-direction (add 10 to avoid conflicts with other boundaries)
      if((std::fabs(cell->face(face_number)->center()(2) + WIDTH_BFS/2.0)< 1.e-12))
        cell->face(face_number)->set_all_boundary_ids (3+10);
    }
  }

  if(use_grid_stretching_in_y_direction == true)
  {
    // manifold
    unsigned int manifold_id = 1;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold;
    triangulation->set_manifold(manifold_id, manifold);
  }

  // periodicity in z-direction
  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 2+10, 3+10, 2, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  // no slip boundaries at the upper and lower wall with ID=0
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // inflow boundary condition at left boundary with ID=2: prescribe velocity profile which
  // is obtained as the results of the simulation on DOMAIN 1
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(2,new InflowProfile<dim>(dim)));

  // outflow boundary condition at right boundary with ID=1
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  // no slip boundaries at the upper and lower wall with ID=0
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // inflow boundary condition at left boundary with ID=2
  // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
  // we assume that this is negligible when using the dual splitting scheme
  boundary_descriptor_pressure->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));

  // outflow boundary condition at right boundary with ID=1: set pressure to zero
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
}


template<int dim>
void set_field_functions_1(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  // use a constant body force for the turbulent channel (DOMAIN 1)
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_field_functions_2(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
//  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  // no body forces for the second domain
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
#include "../../include/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h"
#include "../../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"
#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"

template<int dim>
struct PostProcessorDataBFS
{
  PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
  InflowData<dim> inflow_data;
  LinePlotData<dim> line_plot_data;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PostProcessorBFS : public PostProcessor<dim, degree_u, degree_p, Number>
{
public:
  typedef PostProcessor<dim, degree_u, degree_p, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorBFS(PostProcessorDataBFS<dim> const & pp_data_bfs_in)
    :
    Base(pp_data_bfs_in.pp_data),
    write_final_output(true),
    write_final_output_lines(true),
    pp_data_bfs(pp_data_bfs_in)
  {}

  void setup(NavierStokesOperator const                &navier_stokes_operator_in,
             DoFHandler<dim> const                     &dof_handler_velocity_in,
             DoFHandler<dim> const                     &dof_handler_pressure_in,
             Mapping<dim> const                        &mapping_in,
             MatrixFree<dim,Number> const              &matrix_free_data_in,
             DofQuadIndexData const                    &dof_quad_index_data_in,
             std::shared_ptr<AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    Base::setup(
        navier_stokes_operator_in,
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // turbulent channel statistics for precursor simulation
    statistics_turb_ch.reset(new StatisticsManager<dim>(dof_handler_velocity_in,mapping_in));
    statistics_turb_ch->setup(&grid_transform_turb_channel,pp_data_bfs.turb_ch_data);

    // inflow data
    if(pp_data_bfs.inflow_data.write_inflow_data == true)
    {
      inflow_data_calculator.reset(new InflowDataCalculator<dim,Number>(pp_data_bfs.inflow_data));
      inflow_data_calculator->setup(dof_handler_velocity_in,mapping_in);
    }

    // evaluation of characteristic quantities along lines
    line_plot_calculator_statistics.reset(new LinePlotCalculatorStatisticsHomogeneousDirection<dim>(
        dof_handler_velocity_in, dof_handler_pressure_in, mapping_in));
    line_plot_calculator_statistics->setup(pp_data_bfs.line_plot_data);
  }

  void do_postprocessing(VectorType const &velocity,
                         VectorType const &intermediate_velocity,
                         VectorType const &pressure,
                         double const     time,
                         int const        time_step_number)
  {
    Base::do_postprocessing(
        velocity,
        intermediate_velocity,
        pressure,
        time,
        time_step_number);


    // turbulent channel statistics
    statistics_turb_ch->evaluate(velocity,time,time_step_number);

    // inflow data
    if(pp_data_bfs.inflow_data.write_inflow_data == true)
    {
      inflow_data_calculator->calculate(velocity);
    }

    line_plot_calculator_statistics->evaluate(velocity,pressure,time,time_step_number);
  }

  bool write_final_output;
  bool write_final_output_lines;
  PostProcessorDataBFS<dim> pp_data_bfs;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
  std::shared_ptr<InflowDataCalculator<dim, Number> > inflow_data_calculator;
  std::shared_ptr<LinePlotCalculatorStatisticsHomogeneousDirection<dim> > line_plot_calculator_statistics;
};

template<int dim, int degree_u, int degree_p, typename Number>
std::shared_ptr<PostProcessorBase<dim, degree_u, degree_p, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  PostProcessorDataBFS<dim> pp_data_bfs;
  pp_data_bfs.pp_data = pp_data;
  pp_data_bfs.turb_ch_data = param.turb_ch_data;
  pp_data_bfs.inflow_data = param.inflow_data;
  pp_data_bfs.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessorBase<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorBFS<dim,degree_u,degree_p,Number>(pp_data_bfs));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
