/*
 * periodic_hill.h
 *
 *  Created on: March 28, 2019
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
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY - 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 4;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

bool const WRITE_OUTPUT = true;
std::string OUTPUT_FOLDER = "output/periodic_hill/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";
std::string FILENAME_FLOWRATE = "flow_rate";

// set problem specific parameters like physical dimensions, etc.

//Re = 19000: 0.8284788e-5
//Re = 10595: 1.48571e-5
//Re = 5600:  2.8109185e-5
//Re = 1400:  1.1243674e-4
//Re = 700:   2.2487348e-4
double const RE = 5600.0;
double const VISCOSITY = 2.8109185e-5;

double const H = 0.028;
double const WIDTH = 4.5*H;
double const LENGTH = 9.0*H;
double const HEIGHT = 2.036*H;

double const GRID_STRETCH_FAC = 1.6; // TODO

// RE_H = u_b * H / nu
double const BULK_VELOCITY = RE * VISCOSITY / H;
double const TARGET_FLOW_RATE = BULK_VELOCITY * WIDTH * HEIGHT;

// flow-through time based on bulk velocity
double const FLOW_THROUGH_TIME = LENGTH/BULK_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME = 1.0*FLOW_THROUGH_TIME; // TODO

// sampling
double const SAMPLE_START_TIME = 0.0*FLOW_THROUGH_TIME; // TODO
double const SAMPLE_END_TIME = END_TIME;
unsigned int const SAMPLE_EVERY_TIMESTEPS = 1;
bool const CALCULATE_STATISTICS = true;

QuantityStatistics<DIMENSION> QUANTITY_VELOCITY;
QuantityStatistics<DIMENSION> QUANTITY_REYNOLDS;
//QuantityStatistics<DIMENSION> QUANTITY_PRESSURE;
//QuantityStatisticsPressureCoefficient<DIMENSION> QUANTITY_PRESSURE_COEFF;
//QuantityStatisticsSkinFriction<3> QUANTITY_SKIN_FRICTION;
const unsigned int N_POINTS_LINE = std::pow(2.0,REFINE_STEPS_SPACE_MIN)*(FE_DEGREE_VELOCITY+1)*2;

// data structures that we need to control the mass flow rate:
// NOTA BENE: these variables will be modified by the postprocessor!
double FLOW_RATE = 0.0;
double FLOW_RATE_OLD = 0.0;
// the flow rate controller also needs the time step size as parameter
double TIME_STEP_FLOW_RATE_CONTROLLER = 1.0;

class FlowRateController
{
public:
  FlowRateController()
    :
    // initialize the body force such that the desired flow rate is obtained
    // under the assumption of a parabolic velocity profile in radial direction
    f(0.0), // f(t=t_0) = f_0
    f_damping(0.0)
  {}

  double get_body_force()
  {
    return f + f_damping;
  }

  void update_body_force(unsigned int const time_step_number)
  {
    // use an I-controller with damping (D) to asymptotically reach the desired target flow rate

    // dimensional analysis: [k_I] = 1/(m^2 s^2) -> k_I = const * u_b^2 / H^4
    double const C_I = 100.0;
    double const k_I = C_I * std::pow(BULK_VELOCITY,2.0)/std::pow(H,4.0);
    f += k_I * (TARGET_FLOW_RATE - FLOW_RATE) * TIME_STEP_FLOW_RATE_CONTROLLER;

    // the time step size TIME_STEP_FLOW_RATE_CONTROLLER is 0 when this function is called the first time
    if(time_step_number > 1)
    {
      // dimensional analysis: [k_D] = 1/(m^2) -> k_D = const / H^2
      double const C_D = 0.1;
      double const k_D = C_D / std::pow(H, 2.0);
      f_damping = - k_D * (FLOW_RATE - FLOW_RATE_OLD) / TIME_STEP_FLOW_RATE_CONTROLLER;
    }
  }

private:
  double f;
  double f_damping;
};

// use a global variable which will be called by the postprocessor
// in order to update the body force.
FlowRateController FLOW_RATE_CONTROLLER;

#include "../grid_tools/grid_functions_periodic_hill.h"

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = BULK_VELOCITY;
  cfl = 0.375;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
  order_time_integrator = 2;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  upwind_factor = 0.5;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;

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
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  update_preconditioner_projection = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = WRITE_OUTPUT;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = FLOW_THROUGH_TIME/10.0;
  output_data.write_velocity_magnitude = true;
  output_data.write_vorticity = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_q_criterion = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = FLOW_THROUGH_TIME/10.0;
//  solver_info_data.interval_time_steps = 10;
//  solver_info_data.interval_wall_time = 10.0;

  // calculation of flow rate (use volume-based computation)
  mean_velocity_data.calculate = true;
  mean_velocity_data.filename_prefix = OUTPUT_FOLDER + FILENAME_FLOWRATE;
  Tensor<1,dim,double> direction; direction[0] = 1.0;
  mean_velocity_data.direction = direction;
  mean_velocity_data.write_to_file = true;

  // line plot data: calculate statistics along lines
  line_plot_data.write_output = true;
  line_plot_data.filename_prefix = OUTPUT_FOLDER;
  line_plot_data.statistics_data.calculate_statistics = CALCULATE_STATISTICS;
  line_plot_data.statistics_data.sample_start_time = SAMPLE_START_TIME;
  line_plot_data.statistics_data.sample_end_time = END_TIME;
  line_plot_data.statistics_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
  line_plot_data.statistics_data.write_output_every_timesteps = SAMPLE_EVERY_TIMESTEPS*100;

  // mean velocity
  QUANTITY_VELOCITY.type = QuantityType::Velocity;
  QUANTITY_VELOCITY.average_homogeneous_direction = true;
  QUANTITY_VELOCITY.averaging_direction = 2;

  // Reynolds stresses
  QUANTITY_REYNOLDS.type = QuantityType::ReynoldsStresses;
  QUANTITY_REYNOLDS.average_homogeneous_direction = true;
  QUANTITY_REYNOLDS.averaging_direction = 2;

  // lines
  Line<dim> vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, vel_7, vel_8;

  // begin and end points of all lines
  double const eps = 1.e-10;
  vel_0.begin = Point<dim> (0*H, H+f(0*H)+eps,0);
  vel_0.end =   Point<dim> (0*H, H+HEIGHT-eps,0);
  vel_1.begin = Point<dim> (1*H, H+f(1*H)+eps,0);
  vel_1.end =   Point<dim> (1*H, H+HEIGHT-eps,0);
  vel_2.begin = Point<dim> (2*H, H+f(2*H)+eps,0);
  vel_2.end =   Point<dim> (2*H, H+HEIGHT-eps,0);
  vel_3.begin = Point<dim> (3*H, H+f(3*H)+eps,0);
  vel_3.end =   Point<dim> (3*H, H+HEIGHT-eps,0);
  vel_4.begin = Point<dim> (4*H, H+f(4*H)+eps,0);
  vel_4.end =   Point<dim> (4*H, H+HEIGHT-eps,0);
  vel_5.begin = Point<dim> (5*H, H+f(5*H)+eps,0);
  vel_5.end =   Point<dim> (5*H, H+HEIGHT-eps,0);
  vel_6.begin = Point<dim> (6*H, H+f(6*H)+eps,0);
  vel_6.end =   Point<dim> (6*H, H+HEIGHT-eps,0);
  vel_7.begin = Point<dim> (7*H, H+f(7*H)+eps,0);
  vel_7.end =   Point<dim> (7*H, H+HEIGHT-eps,0);
  vel_8.begin = Point<dim> (8*H, H+f(8*H)+eps,0);
  vel_8.end =   Point<dim> (8*H, H+HEIGHT-eps,0);

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

  // set line names
  vel_0.name = "x_0";
  vel_1.name = "x_1";
  vel_2.name = "x_2";
  vel_3.name = "x_3";
  vel_4.name = "x_4";
  vel_5.name = "x_5";
  vel_6.name = "x_6";
  vel_7.name = "x_7";
  vel_8.name = "x_8";

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
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  Point<dim> p_1;
  p_1[0] = 0.;
  p_1[1] = H;
  if(dim==3)
    p_1[2] = -WIDTH/2.0;

  Point<dim> p_2;
  p_2[0] = LENGTH;
  p_2[1] = H + HEIGHT;
  if (dim == 3)
    p_2[2] = WIDTH/2.0;

  // use 2 cells in x-direction on coarsest grid and 1 cell in y- and z-directions
  std::vector<unsigned int> refinements(dim, 1);
  refinements[0] = 2;
  GridGenerator::subdivided_hyper_rectangle (*triangulation, refinements, p_1, p_2);

  // create hill by shifting y-coordinates of middle vertices by -H in y-direction
  triangulation->last()->vertex(0)[1] = 0.;
  if(dim==3)
    triangulation->last()->vertex(4)[1] = 0.;

  // periodicity in x-direction (add 10 to avoid conflicts with dirichlet boundary, which is 0)
  // make use of the fact that the mesh has only two elements

  // left element
  triangulation->begin()->face(0)->set_all_boundary_ids(0+10);
  // right element
  triangulation->last()->face(1)->set_all_boundary_ids(1+10);

  // periodicity in z-direction
  if(dim == 3)
  {
    // left element
    triangulation->begin()->face(4)->set_all_boundary_ids(2+10);
    triangulation->begin()->face(5)->set_all_boundary_ids(3+10);
    // right element
    triangulation->last()->face(4)->set_all_boundary_ids(4+10);
    triangulation->last()->face(5)->set_all_boundary_ids(5+10);
  }

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 0, periodic_faces);
  if(dim == 3)
  {
    GridTools::collect_periodic_faces(*tria, 2+10, 3+10, 2, periodic_faces);
    GridTools::collect_periodic_faces(*tria, 4+10, 5+10, 2, periodic_faces);
  }

  triangulation->add_periodicity(periodic_faces);

  unsigned int const manifold_id = 111;
  triangulation->begin()->set_all_manifold_ids(manifold_id);
  triangulation->last()->set_all_manifold_ids(manifold_id);

  static const PeriodicHillManifold<dim> manifold;
  triangulation->set_manifold(manifold_id, manifold);

  triangulation->refine_global(n_refine_space);
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h"

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity (const unsigned int n_components = dim,
                           const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int component = 0) const
  {
    double result = 0.0;

    PeriodicHillManifold<dim> manifold;

    // x-velocity
    if(component == 0)
    {
      // initial conditions
      if(p[1] > H && p[1] < (H + HEIGHT))
        result = BULK_VELOCITY * (p[1] - H)*((H + HEIGHT) - p[1])/std::pow(HEIGHT/2.0,2.0);

      // add some random perturbations
      result *= (1.0 + 0.1 * (((double)rand() / RAND_MAX - 0.5)/0.5));
    }

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

 double value (const Point<dim>    & /*p*/,
               const unsigned int  component = 0) const
 {
   double result = 0.0;

   // The flow is driven by constant body force in x-direction
   if(component==0)
   {
     result = FLOW_RATE_CONTROLLER.get_body_force();
   }

   return result;
 }
};

namespace IncNS
{

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
   // set boundary conditions
   typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

   // fill boundary descriptor velocity
   boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

   // fill boundary descriptor pressure
   boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}

// Postprocessor

template<int dim>
struct PostProcessorDataPeriodicHill
{
  PostProcessorData<dim> pp_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotData<dim> line_plot_data;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PostProcessorPeriodicHill : public PostProcessor<dim, degree_u, degree_p, Number>
{
public:
  typedef PostProcessor<dim, degree_u, degree_p, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorPeriodicHill(PostProcessorDataPeriodicHill<dim> const & pp_data_periodic_hill_in)
    :
    Base(pp_data_periodic_hill_in.pp_data),
    pp_data_periodic_hill(pp_data_periodic_hill_in),
    time_old(START_TIME)
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

    // calculation of mean velocity
    mean_velocity_calculator.reset(new MeanVelocityCalculator<dim,degree_u,Number>(
        matrix_free_data_in, dof_quad_index_data_in, pp_data_periodic_hill.mean_velocity_data));

    // evaluation of characteristic quantities along lines
    line_plot_calculator_statistics.reset(new LinePlotCalculatorStatisticsHomogeneousDirection<dim>(
        dof_handler_velocity_in, dof_handler_pressure_in, mapping_in));
    line_plot_calculator_statistics->setup(pp_data_periodic_hill.line_plot_data);
  }

  void do_postprocessing(VectorType const &velocity,
                         VectorType const &pressure,
                         double const     time,
                         int const        time_step_number)
  {
    Base::do_postprocessing(
	      velocity,
        pressure,
        time,
        time_step_number);
   
    if(pp_data_periodic_hill.mean_velocity_data.calculate == true)
    {
      // calculation of flow rate
      FLOW_RATE_OLD = FLOW_RATE;
      FLOW_RATE = mean_velocity_calculator->calculate_flow_rate_volume(velocity, time, LENGTH);

      // set time step size for flow rate controller
      TIME_STEP_FLOW_RATE_CONTROLLER = time-time_old;
      time_old = time;

      // update body force
      FLOW_RATE_CONTROLLER.update_body_force(time_step_number);
    }

    // line plot statistics
    line_plot_calculator_statistics->evaluate(velocity,pressure,time,time_step_number);
  }

private:
  // postprocessor data supplemented with data required for FDA benchmark
  PostProcessorDataPeriodicHill<dim> pp_data_periodic_hill;

  // calculate flow rate in precursor domain so that the flow rate can be
  // dynamically adjusted by a flow rate controller.
  std::shared_ptr<MeanVelocityCalculator<dim,degree_u,Number> > mean_velocity_calculator;

  // the flow rate controller needs the time step size, so we have to store the previous time instant
  double time_old;

  // line plot statistics with averaging in homogeneous direction
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

  PostProcessorDataPeriodicHill<dim> pp_data_periodic_hill;
  pp_data_periodic_hill.pp_data = pp_data;
  pp_data_periodic_hill.mean_velocity_data = param.mean_velocity_data;
  pp_data_periodic_hill.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessorBase<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorPeriodicHill<dim,degree_u,degree_p,Number>(pp_data_periodic_hill));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
