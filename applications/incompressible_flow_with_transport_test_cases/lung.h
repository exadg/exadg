/*
 * lung.h
 *
 *  Created on: March 18, 2019
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

// which lung
#define BABY
//#define ADULT

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;
unsigned int const FE_DEGREE_SCALAR = FE_DEGREE_VELOCITY;

// degree used for mapping
unsigned int const DEGREE_MAPPING = 1;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 0;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// number of lung generations
unsigned int const N_GENERATIONS = 8;

IncNS::TriangulationType const TRIANGULATION_TYPE_FLUID = IncNS::TriangulationType::Distributed;
ConvDiff::TriangulationType const TRIANGULATION_TYPE_SCALAR = ConvDiff::TriangulationType::Distributed;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// output folders
std::string const OUTPUT_FOLDER = "output/lung/";
std::string const OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "5_gen";

// set problem specific parameters
double const VISCOSITY = 1.7e-5; // m^2/s
double const D_OXYGEN = 0.219e-4; // 0.219 cm^2/s = 0.219e-4 m^2/s
double const DENSITY = 1.2; // kg/m^3 (@ 20°C)

#ifdef BABY // preterm infant
double const PERIOD = 0.1; // 100 ms
unsigned int const N_PERIODS = 10;
double const START_TIME = 0.0;
double const END_TIME = PERIOD*N_PERIODS;
double const PEEP_KINEMATIC = 8.0 * 98.0665 / DENSITY; // 8 cmH20, 1 cmH20 = 98.0665 Pa, transform to kinematic pressure
double const TIDAL_VOLUME = 6.6e-6; // 6.6 ml = 6.6 * 10^{-6} m^3
double const C_RS_KINEMATIC = DENSITY * 20.93e-9; // total respiratory compliance C_rs = 20.93 ml/kPa
double const DELTA_P_INITIAL = TIDAL_VOLUME/C_RS_KINEMATIC;
unsigned int const MAX_GENERATION = 24;
// Menache et al. (2008): Extract diameter and length of airways from Table A1 (0.25-year-old female) and compute resistance of airways assuming laminar flow
double const RESISTANCE_VECTOR_DYNAMIC[MAX_GENERATION+1] = // resistance [Pa/(m^3/s)]
{
    9.59E+03, // GENERATION 0
    1.44E+04,
    3.66E+04,
    1.37E+05,
    5.36E+05,
    1.78E+06,
    4.36E+06,
    1.13E+07,
    2.60E+07,
    4.30E+07,
    8.46E+07,
    1.38E+08,
    2.29E+08,
    3.06E+08,
    3.64E+08,
    6.24E+08,
    9.02E+08,
    1.08E+09,
    1.36E+09,
    1.75E+09,
    2.41E+09,
    3.65E+09,
    3.45E+09,
    5.54E+09,
    1.62E+09 // MAX_GENERATION
};
#endif
#ifdef ADULT // adult lung
double const PERIOD = 3; // 3 s
unsigned int const N_PERIODS = 10;
double const START_TIME = 0.0;
double const END_TIME = PERIOD*N_PERIODS;
double const PEEP_KINEMATIC = 8.0 * 98.0665 / DENSITY; // 8 cmH20, 1 cmH20 = 98.0665 Pa, transform to kinematic pressure
double const TIDAL_VOLUME = 500.0e-6; // 500 ml = 500 * 10^{-6} m^3
double const C_RS_KINEMATIC = DENSITY * 100.0e-6/98.0665; // total respiratory compliance C_rs = 100 ml/cm H20
double const DELTA_P_INITIAL = TIDAL_VOLUME/C_RS_KINEMATIC;
unsigned int const MAX_GENERATION = 25;
// Menache et al. (2008): Extract diameter and length of airways from Table A11 (21-year-old male) and compute resistance of airways assuming laminar flow
double const RESISTANCE_VECTOR_DYNAMIC[MAX_GENERATION+1] = // resistance [Pa/(m^3/s)]
{
    5.96E+02, // GENERATION 0
    3.87E+02,
    1.06E+03,
    2.57E+03,
    7.93E+03,
    3.04E+04,
    7.82E+04,
    2.35E+05,
    5.60E+05,
    1.50E+06,
    2.06E+06,
    3.24E+06,
    4.57E+06,
    6.38E+06,
    8.53E+06,
    1.11E+07,
    1.58E+07,
    2.08E+07,
    2.62E+07,
    3.39E+07,
    4.11E+07,
    5.04E+07,
    5.61E+07,
    6.34E+07,
    7.11E+07,
    4.73E+07 // MAX_GENERATION
};
#endif

double const CFL_OIF = 0.35;
double const CFL = CFL_OIF;
double const MAX_VELOCITY = 3.0;
bool const ADAPTIVE_TIME_STEPPING = true;

// solver tolerances
double const ABS_TOL = 1.e-12;
double const REL_TOL = 1.e-3;

// outlet boundary IDs
types::boundary_id const OUTLET_ID_FIRST = 2;
types::boundary_id OUTLET_ID_LAST = 2;

// output
bool const WRITE_OUTPUT = true;
double const OUTPUT_START_TIME = START_TIME;
double const OUTPUT_INTERVAL_TIME = PERIOD/20;

// restart
bool const WRITE_RESTART = false;
double const RESTART_INTERVAL_TIME = PERIOD;

double get_equivalent_resistance()
{
  double resistance = 0.0;

  // calculate effective resistance for all higher generations not being resolved
  // assuming that all airways of a specific generation have the same resistance and that the flow
  // is laminar!
  for(unsigned int i = 0; i <= MAX_GENERATION - N_GENERATIONS; ++i)
  {
    resistance += RESISTANCE_VECTOR_DYNAMIC[i + N_GENERATIONS]/std::pow(2.0, (double)i);
  }

  // beyond the current outflow boundary, we have two branches from generation N_GENERATIONS to MAX_GENERATION
  resistance /= 2.0;

  // the solver uses the kinematic pressure and therefore we have to transform the resistance
  resistance /= DENSITY;

  return resistance;
}

template<int dim>
void
IncNS::InputParameters<dim>::
set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  use_outflow_bc_convective_term = true;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  time_integrator_oif = TimeIntegratorOIF::ExplRK2Stage2;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  time_step_size_max = 5.e-5;
  max_velocity = MAX_VELOCITY;
  cfl_oif = CFL_OIF;
  cfl = CFL;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-3;
  order_time_integrator = 2;
  start_with_low_order = true;

  // NUMERICAL PARAMETERS
  implement_block_diagonal_preconditioner_matrix_free = false;
  use_cell_based_face_loops = false;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TRIANGULATION_TYPE_FLUID;

  // mapping
  degree_mapping = DEGREE_MAPPING;

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
  multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;

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
  output_data.write_output = WRITE_OUTPUT;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME + "_fluid";
  output_data.output_start_time = OUTPUT_START_TIME;
  output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
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

  // calculation of flow rate
  flow_rate_data.calculate = true;
  flow_rate_data.write_to_file = true;
  flow_rate_data.filename_prefix = OUTPUT_FOLDER + "flow_rate";
  // Note: The set with boundary IDs is filled later once we now the grid and the outlet boundaries

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = PERIOD/200;
}

void
ConvDiff::InputParameters::
set_input_parameters(unsigned int scalar_index)
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::ConvectionDiffusion;
  type_velocity_field = TypeVelocityField::Numerical;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  if(scalar_index == 0)
  {
    diffusivity = D_OXYGEN;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDF;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  order_time_integrator = 2;
  time_integrator_oif = TimeIntegratorRK::ExplRK3Stage7Reg2;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  time_step_size = 1.0e-2;
  cfl_oif = CFL_OIF;
  cfl = CFL;
  max_velocity = MAX_VELOCITY;
  exponent_fe_degree_convection = 1.5;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TRIANGULATION_TYPE_SCALAR;

  // mapping
  degree_mapping = DEGREE_MAPPING;

  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::GMRES;
  solver_data = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner = Preconditioner::InverseMassMatrix; //BlockJacobi; //Multigrid;
  implement_block_diagonal_preconditioner_matrix_free = false;
  use_cell_based_face_loops = false;
  update_preconditioner = false;

  multigrid_data.type = MultigridType::hMG;
  mg_operator_type = MultigridOperatorType::ReactionConvectionDiffusion;
  // MG smoother
  multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;
  // MG smoother data
  multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data.smoother_data.iterations = 5;

  // MG coarse grid solver
  multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  output_data.write_output = WRITE_OUTPUT;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
  output_data.output_start_time = OUTPUT_START_TIME;
  output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  output_data.degree = FE_DEGREE_SCALAR;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // restart
  restart_data.write_restart = WRITE_RESTART;
  restart_data.interval_time = RESTART_INTERVAL_TIME;
  restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
}

/*
 * This class controls the pressure at the inlet to obtain a desired tidal volume
 */
class Ventilator
{
public:
  Ventilator()
    :
      pressure_difference(DELTA_P_INITIAL),
      pressure_difference_last_period(DELTA_P_INITIAL),
      pressure_difference_damping(0.0),
      volume_max(std::numeric_limits<double>::min()),
      volume_min(std::numeric_limits<double>::max()),
      tidal_volume_last(TIDAL_VOLUME),
      C_I(0.4), // choose C_I = 0.1-1.0 (larger value might improve speed of convergence to desired value; instabilities detected for C_I = 1 and larger)
      C_D(C_I*0.2),
      counter(0),
      counter_last(0)
  {}

  double
  get_pressure(double const &time) const
  {
    // 0 <= (t-t_period_start) <= PERIOD/3 (inhaling)
    if((int(time/(PERIOD/3)))%3 == 0)
    {
      return PEEP_KINEMATIC + pressure_difference + pressure_difference_damping;
    }
    else // rest of the period (exhaling)
    {
      return PEEP_KINEMATIC;
    }
  }

  void
  update_pressure_difference(double const time, double const volume)
  {
    // always update volumes
    volume_max = std::max(volume, volume_max);
    volume_min = std::min(volume, volume_min);

    // recalculate pressure difference only once every period
    if(new_period(time))
    {
      if(counter >= 1)
      {
        recalculate_pressure_difference();
      }

      // reset volumes
      volume_max = std::numeric_limits<double>::min();
      volume_min = std::numeric_limits<double>::max();
    }
  }

private:
  bool
  new_period(double const time)
  {
    counter = int(time/PERIOD);
    if(counter > counter_last)
    {
      counter_last = counter;
      return true;
    }
    else
    {
      return false;
    }
  }

  void
  recalculate_pressure_difference()
  {
    pressure_difference = pressure_difference_last_period + C_I * (TIDAL_VOLUME - (volume_max-volume_min))/TIDAL_VOLUME * PEEP_KINEMATIC; // I-controller

    if(counter >= 2)
      pressure_difference_damping = - C_D * ((volume_max-volume_min) - tidal_volume_last)/TIDAL_VOLUME * PEEP_KINEMATIC; // D-controller
    else
      pressure_difference_damping = 0.0;

    pressure_difference_last_period = pressure_difference;
    tidal_volume_last = volume_max-volume_min;
  }

  double pressure_difference;
  double pressure_difference_last_period;
  double pressure_difference_damping;
  double volume_max;
  double volume_min;
  double tidal_volume_last;
  double const C_I, C_D;
  unsigned int counter;
  unsigned int counter_last;
};

std::shared_ptr<Ventilator> VENTILATOR;

template<int dim>
class PressureInlet : public Function<dim>
{
public:
  PressureInlet (std::shared_ptr<Ventilator> ventilator_,
                 const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time),
    ventilator(ventilator_)
  {}

  double value (const Point<dim>   &/*p*/,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    double pressure = ventilator->get_pressure(t);

    return pressure;
  }

private:
  std::shared_ptr<Ventilator> ventilator;
};

class OutflowBoundary
{
public:
  OutflowBoundary(types::boundary_id const id)
    :
      boundary_id(id),
      resistance(get_equivalent_resistance()), // in preliminary tests with 5 generations we used a constant value of 1.0e7
      compliance(C_RS_KINEMATIC/std::pow(2.0, N_GENERATIONS-1)), // TODO one could use statistical distribution as in Roth et al. (2018)
      volume(compliance * PEEP_KINEMATIC), // p = 1/C * V -> V = C * p (initialize volume so that p(t=0) = PEEP_KINEMATIC)
      flow_rate(0.0),
      time_old(START_TIME)
  {}

  void
  set_flow_rate(double const flow_rate_)
  {
    flow_rate = flow_rate_;
  }

  void
  integrate_volume(double const time)
  {
    // currently use BDF1 time integration // TODO one could use a higher order time integrator
    volume += flow_rate*(time-time_old);
    time_old = time;
  }

  double
  get_pressure() const
  {
    return resistance*flow_rate + volume/compliance;
  }

  double
  get_volume() const
  {
    return volume;
  }

  types::boundary_id get_boundary_id() const
  {
    return boundary_id;
  }

private:
  types::boundary_id const boundary_id;
  double resistance;
  double compliance;
  double volume;
  double flow_rate;
  double time_old;
};

// we need individual outflow boundary conditions for each outlet
std::vector<std::shared_ptr<OutflowBoundary>> OUTFLOW_BOUNDARIES;
// we need to compute the flow rate for each outlet
std::map<types::boundary_id, double> FLOW_RATES;

template<int dim>
class PressureOutlet : public Function<dim>
{
public:
  PressureOutlet (std::shared_ptr<OutflowBoundary> outflow_boundary_,
                  double const time = 0.)
    :
    Function<dim>(1 /*n_components*/, time),
    outflow_boundary(outflow_boundary_)
  {}

  double value (const Point<dim>   &/*p*/,
                const unsigned int /*component*/) const
  {
    return outflow_boundary->get_pressure();
  }

private:
  std::shared_ptr<OutflowBoundary> outflow_boundary;
};

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
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  std::vector<std::string> files;
  get_lung_files_from_environment(files);
  auto tree_factory = dealii::GridGenerator::lung_files_to_node(files);
  std::string spline_file = get_lung_spline_file_from_environment();

  std::map<std::string, double> timings;
  
  std::shared_ptr<LungID::Checker> generation_limiter(new LungID::GenerationChecker(N_GENERATIONS));
  //std::shared_ptr<LungID::Checker> generation_limiter(new LungID::ManualChecker());

  // create triangulation
  if(auto tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
  {
    dealii::GridGenerator::lung(*tria,
                                n_refine_space,
                                n_refine_space,
                                tree_factory,
                                timings,
                                OUTLET_ID_FIRST,
                                OUTLET_ID_LAST,
                                spline_file,
                                generation_limiter);
  }
  else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
  {
    dealii::GridGenerator::lung(*tria,
                                n_refine_space,
                                tree_factory,
                                timings,
                                OUTLET_ID_FIRST,
                                OUTLET_ID_LAST,
                                spline_file,
                                generation_limiter);
  }
  else
  {
    AssertThrow(false, ExcMessage("Unknown triangulation!"));
  }

  //AssertThrow(OUTLET_ID_LAST-OUTLET_ID_FIRST == std::pow(2, N_GENERATIONS - 1), ExcMessage("Number of outlets has to be 2^{N_generations-1}."));
}

/**************************************************************************************/
/*                                                                                    */
/*          FUNCTIONS (ANALYTICAL/INITIAL SOLUTION, BOUNDARY CONDITIONS, etc.)        */
/*                                                                                    */
/**************************************************************************************/

namespace IncNS
{

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  // set boundary conditions
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // 0 = walls
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

  // 1 = inlet
  VENTILATOR.reset(new Ventilator());
  boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new PressureInlet<dim>(VENTILATOR)));

  // outlets
  for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
  {
    std::shared_ptr<OutflowBoundary> outflow_boundary;
    outflow_boundary.reset(new OutflowBoundary(id));
    OUTFLOW_BOUNDARIES.push_back(outflow_boundary);

    boundary_descriptor_velocity->neumann_bc.insert(pair(id, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(id, new PressureOutlet<dim>(outflow_boundary)));
  }
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


template<int dim>
struct PostProcessorDataLung
{
  PostProcessorData<dim> pp_data;
  FlowRateCalculatorData<dim> flow_rate_data;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PostProcessorLung : public PostProcessor<dim, degree_u, degree_p, Number>
{
public:
  typedef PostProcessor<dim, degree_u, degree_p, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorLung(PostProcessorDataLung<dim> const & pp_data_in)
    :
    Base(pp_data_in.pp_data),
    pp_data_lung(pp_data_in),
    time_last(START_TIME)
  {
  }

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

    // fill FLOW_RATES map
    for(auto iterator = OUTFLOW_BOUNDARIES.begin(); iterator != OUTFLOW_BOUNDARIES.end(); ++iterator)
    {
      FLOW_RATES.insert(std::pair<types::boundary_id, double>((*iterator)->get_boundary_id(),0.0));
    }

    // flow rates:
    // In a first step, fill the set with boundary IDs. Note that we have to do that here and cannot do this step
    // in the function set_input_parameters() since the list of outflow boundaries is not known at that time.
    // The list of outflow boundaries is known once the grid has been created and the outflow boundary IDs have been set.
    for(auto iterator = OUTFLOW_BOUNDARIES.begin(); iterator != OUTFLOW_BOUNDARIES.end(); ++iterator)
    {
      pp_data_lung.flow_rate_data.boundary_IDs.insert((*iterator)->get_boundary_id());
    }

    flow_rate_calculator.reset(new FlowRateCalculator<dim,degree_u,Number>(
        matrix_free_data_in, dof_handler_velocity_in, dof_quad_index_data_in, pp_data_lung.flow_rate_data));
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

    // calculate flow rates for all outflow boundaries
    AssertThrow(pp_data_lung.flow_rate_data.calculate == true, ExcMessage("Activate flow rate computation."));
    flow_rate_calculator->calculate_flow_rates(velocity, time, FLOW_RATES);

    // set flow rate for all outflow boundaries and update volume (i.e., integrate flow rate over time)
    Number volume = 0.0;
    for(auto iterator = OUTFLOW_BOUNDARIES.begin(); iterator != OUTFLOW_BOUNDARIES.end(); ++iterator)
    {
      (*iterator)->set_flow_rate(FLOW_RATES.at((*iterator)->get_boundary_id()));
      (*iterator)->integrate_volume(time);
      volume += (*iterator)->get_volume();
    }

    // write volume to file
    if(pp_data_lung.flow_rate_data.write_to_file)
    {
      std::ostringstream filename;
      filename << OUTPUT_FOLDER + "volume";
      write_output(volume, time, "Volume in [m^3]", time_step_number, filename);

      // write time step size
      std::ostringstream filename_dt;
      filename_dt << OUTPUT_FOLDER + "time_step_size";
      write_output(time-time_last, time, "Time step size in [s]", time_step_number, filename_dt);
      time_last = time;
    }

    // update the ventilator using the new volume
    VENTILATOR->update_pressure_difference(time, volume);

    // write pressure to file
    if(pp_data_lung.flow_rate_data.write_to_file)
    {
      double const pressure = VENTILATOR->get_pressure(time);
      std::ostringstream filename;
      filename << OUTPUT_FOLDER + "pressure";
      write_output(pressure, time, "Pressure in [m^2/s^2]", time_step_number, filename);
    }
  }

private:
  void
  write_output(double const & value, double const & time, std::string const & name, unsigned int const time_step_number, std::ostringstream const &filename)
  {
    // write output file
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream f;
      if(time_step_number == 1)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
        f << std::endl << "  Time                " + name << std::endl;
      }
      else
      {
        f.open(filename.str().c_str(), std::ios::app);
      }

      unsigned int precision = 12;
      f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
        << std::setw(precision + 8) << value << std::endl;
    }
  }

  // postprocessor data supplemented with data required for lung
  PostProcessorDataLung<dim> pp_data_lung;

  // calculate flow rates for all outflow boundaries
  std::shared_ptr<FlowRateCalculator<dim,degree_u,Number> > flow_rate_calculator;

  double time_last;
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
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.kinetic_energy_spectrum_data = param.kinetic_energy_spectrum_data;

  // Lung specific modules
  PostProcessorDataLung<dim> pp_data_lung;
  pp_data_lung.pp_data = pp_data;
  pp_data_lung.flow_rate_data = param.flow_rate_data;

  std::shared_ptr<PostProcessorLung<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorLung<dim,degree_u,degree_p,Number>(pp_data_lung));

  return pp;
}

}

#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"

namespace ConvDiff
{

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC (const unsigned int n_components = 1,
               const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  /*component = 0*/) const
  {
    return 1.0;
  }
};

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // 0 = walls
  boundary_descriptor->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));

  // 1 = inlet
  boundary_descriptor->dirichlet_bc.insert(pair(1,new DirichletBC<dim>()));

  // outlets
  for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
  {
    // TODO
    boundary_descriptor->dirichlet_bc.insert(pair(id, new Functions::ZeroFunction<dim>(1)));
  }
}

template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  field_functions->analytical_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  analytical_solution->solution.reset(new Functions::ZeroFunction<dim>(1));
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
