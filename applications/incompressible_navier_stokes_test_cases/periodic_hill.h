/*
 * periodic_hill.h
 *
 *  Created on: March 28, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h"
#include "../../include/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

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

QuantityStatistics<3> QUANTITY_VELOCITY;
QuantityStatistics<3> QUANTITY_REYNOLDS;
//QuantityStatistics<DIMENSION> QUANTITY_PRESSURE;
//QuantityStatisticsPressureCoefficient<DIMENSION> QUANTITY_PRESSURE_COEFF;
//QuantityStatisticsSkinFriction<3> QUANTITY_SKIN_FRICTION;
const unsigned int N_POINTS_LINE = std::pow(2.0,REFINE_SPACE_MIN)*(DEGREE_MIN+1)*2;

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

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = true;


  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = BULK_VELOCITY;
  param.cfl = 0.375;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = FLOW_THROUGH_TIME/10.0;
//  param.solver_info_data.interval_time_steps = 10;
//  param.solver_info_data.interval_wall_time = 10.0;


  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  param.upwind_factor = 0.5;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;

  // TURBULENCE
  param.use_turbulence_model = false;
  param.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165
  // Vreman: 0.28
  // WALE: 0.50
  // Sigma: 1.35
  param.turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  param.update_preconditioner_projection = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

#include "../grid_tools/grid_functions_periodic_hill.h"

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
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

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
struct PostProcessorDataPeriodicHill
{
  PostProcessorData<dim> pp_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
};

template<int dim, typename Number>
class PostProcessorPeriodicHill : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorPeriodicHill(PostProcessorDataPeriodicHill<dim> const & pp_data_periodic_hill_in)
    :
    Base(pp_data_periodic_hill_in.pp_data),
    pp_data_periodic_hill(pp_data_periodic_hill_in),
    time_old(START_TIME)
  {}

  void setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // calculation of mean velocity
    mean_velocity_calculator.reset(new MeanVelocityCalculator<dim,Number>(
        pde_operator.get_matrix_free(),
        pde_operator.get_dof_index_velocity(),
        pde_operator.get_quad_index_velocity_linear(),
        pp_data_periodic_hill.mean_velocity_data));

    // evaluation of characteristic quantities along lines
    line_plot_calculator_statistics.reset(new LinePlotCalculatorStatisticsHomogeneousDirection<dim>(
        pde_operator.get_dof_handler_u(),
        pde_operator.get_dof_handler_p(),
        pde_operator.get_mapping()));

    line_plot_calculator_statistics->setup(pp_data_periodic_hill.pp_data.line_plot_data);
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
  std::shared_ptr<MeanVelocityCalculator<dim,Number> > mean_velocity_calculator;

  // the flow rate controller needs the time step size, so we have to store the previous time instant
  double time_old;

  // line plot statistics with averaging in homogeneous direction
  std::shared_ptr<LinePlotCalculatorStatisticsHomogeneousDirection<dim> > line_plot_calculator_statistics;
};

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = WRITE_OUTPUT;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = FLOW_THROUGH_TIME/10.0;
  pp_data.output_data.write_velocity_magnitude = true;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_vorticity_magnitude = true;
  pp_data.output_data.write_q_criterion = true;
  pp_data.output_data.degree = param.degree_u;

  // line plot data: calculate statistics along lines
  pp_data.line_plot_data.write_output = true;
  pp_data.line_plot_data.filename_prefix = OUTPUT_FOLDER;
  pp_data.line_plot_data.statistics_data.calculate_statistics = CALCULATE_STATISTICS;
  pp_data.line_plot_data.statistics_data.sample_start_time = SAMPLE_START_TIME;
  pp_data.line_plot_data.statistics_data.sample_end_time = END_TIME;
  pp_data.line_plot_data.statistics_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
  pp_data.line_plot_data.statistics_data.write_output_every_timesteps = SAMPLE_EVERY_TIMESTEPS*100;

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
  pp_data.line_plot_data.lines.push_back(vel_0);
  pp_data.line_plot_data.lines.push_back(vel_1);
  pp_data.line_plot_data.lines.push_back(vel_2);
  pp_data.line_plot_data.lines.push_back(vel_3);
  pp_data.line_plot_data.lines.push_back(vel_4);
  pp_data.line_plot_data.lines.push_back(vel_5);
  pp_data.line_plot_data.lines.push_back(vel_6);
  pp_data.line_plot_data.lines.push_back(vel_7);
  pp_data.line_plot_data.lines.push_back(vel_8);

  PostProcessorDataPeriodicHill<dim> pp_data_periodic_hill;
  pp_data_periodic_hill.pp_data = pp_data;

  // calculation of flow rate (use volume-based computation)
  pp_data_periodic_hill.mean_velocity_data.calculate = true;
  pp_data_periodic_hill.mean_velocity_data.filename_prefix = OUTPUT_FOLDER + FILENAME_FLOWRATE;
  Tensor<1,dim,double> direction; direction[0] = 1.0;
  pp_data_periodic_hill.mean_velocity_data.direction = direction;
  pp_data_periodic_hill.mean_velocity_data.write_to_file = true;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessorPeriodicHill<dim,Number>(pp_data_periodic_hill));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
