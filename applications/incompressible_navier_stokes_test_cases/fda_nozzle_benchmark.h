/*
 * fda_nozzle_benchmark.h
 *
 *  Created on: May, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"
#include "../../include/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h"
#include "../../include/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h"
#include "../../include/functionalities/linear_interpolation.h"

// nozzle geometry
#include "../grid_tools/fda_benchmark/nozzle.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = DEGREE_MIN;

unsigned int const REFINE_SPACE_MIN = 1;
unsigned int const REFINE_SPACE_MAX = REFINE_SPACE_MIN;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters

// space dimensions
unsigned int const DIMENSION = 3;

// polynomial degree (velocity)
unsigned int const DEGREE_U = DEGREE_MIN;

// set the number of refine levels for DOMAIN 1
unsigned int const REFINE_STEPS_SPACE_DOMAIN1 = REFINE_SPACE_MIN + 1;

// set the number of refine levels for DOMAIN 2
unsigned int const REFINE_STEPS_SPACE_DOMAIN2 = REFINE_SPACE_MIN;

// prescribe velocity inflow profile for nozzle domain via precursor simulation?
// USE_PRECURSOR_SIMULATION == true:  use solver incompressible_navier_stokes_two_domains.cc
// USE_PRECURSOR_SIMULATION == false: use solver incompressible_navier_stokes.cc
bool const USE_PRECURSOR_SIMULATION = true;

// use prescribed velocity profile at inflow superimposed by random perturbations (white noise)?
// This option is only relevant if USE_PRECURSOR_SIMULATION == false
bool const USE_RANDOM_PERTURBATION = false;
// amplitude of perturbations relative to maximum velocity on centerline
double const FACTOR_RANDOM_PERTURBATIONS = 0.02;

// set the throat Reynolds number Re_throat = U_{mean,throat} * (2 R_throat) / nu
double const RE = 3500; //500; //2000; //3500; //5000; //6500; //8000;

// output folders
std::string OUTPUT_FOLDER = "output/fda/Re3500/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_1 = "precursor";
std::string OUTPUT_NAME_2 = "nozzle";
std::string FILENAME_FLOWRATE = "precursor_mean_velocity";

// set problem specific parameters like physical dimensions, etc.

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
bool const WRITE_OUTPUT = false;
double const OUTPUT_START_TIME_PRECURSOR = START_TIME_PRECURSOR;
double const OUTPUT_START_TIME_NOZZLE = START_TIME_NOZZLE;
double const OUTPUT_INTERVAL_TIME = 5.0*T_0;  //10.0*T_0;

// sampling

// sampling interval should last over (100-200) * T_0 according to preliminary results.
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

// data structures that we need in order to apply the velocity inflow profile:

// - we currently use global variables for this purpose
// - choose a large number of points to ensure a smooth inflow profile
unsigned int N_POINTS_R = 10 * (DEGREE_U+1) * std::pow(2.0, REFINE_STEPS_SPACE_DOMAIN1);
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

  // 0 <= radius <= R_OUTER
  for(unsigned int i=0; i<N_POINTS_R; ++i)
    R_VALUES[i] = double(i)/double(N_POINTS_R-1)*R_OUTER;

  // - pi <= phi <= pi
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

      VELOCITY_VALUES[iy*N_POINTS_PHI + iz] = velocity;
    }
  }
}

void add_random_perturbations()
{
  AssertThrow(N_POINTS_R >= 2, ExcMessage("Variable N_POINTS_R is invalid"));
  AssertThrow(N_POINTS_PHI >= 2, ExcMessage("Variable N_POINTS_PHI is invalid"));

  for(unsigned int iy=0; iy<N_POINTS_R; ++iy)
  {
    for(unsigned int iz=0; iz<N_POINTS_PHI; ++iz)
    {
      // Add random perturbation
      double perturbation = FACTOR_RANDOM_PERTURBATIONS * ((double)rand()/RAND_MAX-0.5)/0.5;

      VELOCITY_VALUES[iy*N_POINTS_PHI + iz] *= (1.0 + perturbation);
    }
  }
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
namespace IncNS
{
void set_input_parameters(InputParameters &param, unsigned int const domain_id)
{
  // MATHEMATICAL MODEL
  param.dim = DIMENSION;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.use_outflow_bc_convective_term = true;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = true;


  // PHYSICAL QUANTITIES
  if(domain_id == 1)
    param.start_time = START_TIME_PRECURSOR;
  else if(domain_id == 2)
    param.start_time = START_TIME_NOZZLE;

  param.end_time = END_TIME;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;

//  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
//  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
//  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
//  param.adaptive_time_stepping = true;
  param.temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping_limiting_factor = 3.0;
  param.max_velocity = MAX_VELOCITY_CFL;
  param.cfl = 4.0;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.dt_refinements = 0;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = T_0;


  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_U;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;

  if(domain_id == 1)
    param.h_refinements = REFINE_STEPS_SPACE_DOMAIN1;
  else if(domain_id == 2)
    param.h_refinements = REFINE_STEPS_SPACE_DOMAIN2;

  // convective term
  param.upwind_factor = 1.0;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  param.IP_factor_viscous = 1.0;

  // special case: pure DBC's
  if(domain_id == 1)
    param.pure_dirichlet_bc = true;
  else if(domain_id == 2)
    param.pure_dirichlet_bc = false;

  // div-div and continuity penalty terms
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.add_penalty_terms_to_monolithic_system = false;

  // TURBULENCE
  param.use_turbulence_model = false;
  param.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165, Vreman: 0.28, WALE: 0.50, Sigma: 1.35
  param.turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.IP_factor_pressure = 1.0;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-3,100);
  param.solver_pressure_poisson = SolverPressurePoisson::CG; //FGMRES;
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  if(domain_id == 1)
    param.multigrid_data_pressure_poisson.type = MultigridType::phMG;
  else if(domain_id == 2)
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;


  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-3);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  param.update_preconditioner_projection = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-3);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
  param.rotational_formulation = true; // use false for standard formulation

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-3);

  // linear solver
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-1, 100);
  else
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-3, 100);

  param.solver_momentum = SolverMomentum::GMRES;
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;

  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-20,1.e-3);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES; //GMRES; //FGMRES;
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-1, 100);
  else
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 100);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // Chebyshev moother
  param.multigrid_data_pressure_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
}

// solve problem for DOMAIN 2 only (nozzle domain)
void set_input_parameters(InputParameters & param)
{
  // call set_input_parameters() function for DOMAIN 2
  set_input_parameters(param, 2);
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

/*
 *  Create grid for precursor domain (DOMAIN 1)
 */
template<int dim>
void create_grid_and_set_boundary_ids_1(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /*
   *   PRECURSOR
   */
  Triangulation<2> tria_2d;
  GridGenerator::hyper_ball(tria_2d, Point<2>(), R_OUTER);
  GridGenerator::extrude_triangulation(tria_2d,N_CELLS_AXIAL_PRECURSOR+1,LENGTH_PRECURSOR,*triangulation);
  Tensor<1,dim> offset = Tensor<1,dim>();
  offset[2] = Z1_PRECURSOR;
  GridTools::shift(offset,*triangulation);

  /*
   *  MANIFOLDS
   */
  triangulation->set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin();cell != triangulation->end(); ++cell)
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
    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],Point<dim>())));
        triangulation->set_manifold(manifold_ids[i],*(manifold_vec[i]));
      }
    }
  }

  /*
   *  BOUNDARY ID's
   */
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
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

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 2, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);
}

/*
 *  Create grid for precursor domain (DOMAIN 2)
 */
template<int dim>
void create_grid_and_set_boundary_ids_2(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  create_grid_and_set_boundary_ids_nozzle(triangulation, n_refine_space, periodic_faces);
}

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  // call respective function for DOMAIN 2
  create_grid_and_set_boundary_ids_2(triangulation,
                                     n_refine_space,
                                     periodic_faces);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

namespace IncNS
{

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

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
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
};

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

  double value (const Point<dim>   &p,
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

 double value (const Point<dim>    & /*p*/,
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

template<int dim>
void set_boundary_conditions_1(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  // no slip boundaries at lower and upper wall with ID=0
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  // no slip boundaries at lower and upper wall with ID=0
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
}

template<int dim>
void set_boundary_conditions_2(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  // no slip boundaries at the upper and lower wall with ID=0
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // inflow boundary condition at left boundary with ID=1: prescribe velocity profile which
  // is obtained as the results of the simulation on DOMAIN 1
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new InflowProfile<dim>(dim)));

  // outflow boundary condition at right boundary with ID=2
  boundary_descriptor_velocity->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  // no slip boundaries at the upper and lower wall with ID=0
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));

  // inflow boundary condition at left boundary with ID=1
  // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
  // we assume that this is negligible when using the dual splitting scheme
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

  // outflow boundary condition at right boundary with ID=2: set pressure to zero
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(2,new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  set_boundary_conditions_2(boundary_descriptor_velocity,
                            boundary_descriptor_pressure);
}

template<int dim>
void set_field_functions_1(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  // prescribe body force for the turbulent channel (DOMAIN 1) to adjust the desired flow rate
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_field_functions_2(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // call respective function for DOMAIN 2
  set_field_functions_2(field_functions);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
struct PostProcessorDataFDA
{
  PostProcessorData<dim> pp_data;
  InflowData<dim> inflow_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotData<dim> line_plot_data;
};

template<int dim, typename Number>
class PostProcessorFDA : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorFDA(PostProcessorDataFDA<dim> const & pp_data_in)
    :
    Base(pp_data_in.pp_data),
    pp_data_fda(pp_data_in),
    time_old(START_TIME_PRECURSOR)
  {
    inflow_data_calculator.reset(new InflowDataCalculator<dim,Number>(pp_data_in.inflow_data));
  }

  void setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // inflow data
    inflow_data_calculator->setup(pde_operator.get_dof_handler_u(),
                                  pde_operator.get_mapping());

    // calculation of mean velocity
    mean_velocity_calculator.reset(new MeanVelocityCalculator<dim,Number>(
        pde_operator.get_matrix_free(),
        pde_operator.get_dof_index_velocity(),
        pde_operator.get_quad_index_velocity_linear(),
        pp_data_fda.mean_velocity_data));

    // evaluation of results along lines
    line_plot_calculator_statistics.reset(new LinePlotCalculatorStatistics<dim>(
        pde_operator.get_dof_handler_u(),
        pde_operator.get_dof_handler_p(),
        pde_operator.get_mapping()));

    line_plot_calculator_statistics->setup(pp_data_fda.line_plot_data);
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

    if(USE_PRECURSOR_SIMULATION == true)
    {
      // inflow data
      inflow_data_calculator->calculate(velocity);

      // random perturbations
      if(USE_RANDOM_PERTURBATION==true)
        add_random_perturbations();
    }
    else // laminar inflow profile
    {
      // in case of random perturbations, the velocity field at the inflow boundary
      // has to be recomputed after each time step
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
  std::shared_ptr<MeanVelocityCalculator<dim, Number> > mean_velocity_calculator;

  // the flow rate controller needs the time step size, so we have to store the previous time instant
  double time_old;

  // evaluation of results along lines
  std::shared_ptr<LinePlotCalculatorStatistics<dim> > line_plot_calculator_statistics;
};


template<int dim>
void
initialize_postprocessor_data(PostProcessorDataFDA<dim> &pp_data_fda,
                              InputParameters const     &param,
                              unsigned int const        domain_id)
{
  (void)param;

  // basic modules
  PostProcessorData<dim> pp_data;

  if(domain_id == 1)
  {
    // write output for visualization of results
    pp_data.output_data.write_output = WRITE_OUTPUT;
    pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
    pp_data.output_data.output_name = OUTPUT_NAME_1;
    pp_data.output_data.output_start_time = OUTPUT_START_TIME_PRECURSOR;
    pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.mean_velocity.calculate = true;
    pp_data.output_data.mean_velocity.sample_start_time = SAMPLE_START_TIME;
    pp_data.output_data.mean_velocity.sample_end_time = SAMPLE_END_TIME;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.degree = param.degree_u;

    pp_data_fda.pp_data = pp_data;

    // inflow data
    // prescribe solution at the right boundary of the precursor domain
    // as weak Dirichlet boundary condition at the left boundary of the nozzle domain
    pp_data_fda.inflow_data.write_inflow_data = true;
    pp_data_fda.inflow_data.inflow_geometry = InflowGeometry::Cylindrical;
    pp_data_fda.inflow_data.normal_direction = 2;
    pp_data_fda.inflow_data.normal_coordinate = Z2_PRECURSOR;
    pp_data_fda.inflow_data.n_points_y = N_POINTS_R;
    pp_data_fda.inflow_data.n_points_z = N_POINTS_PHI;
    pp_data_fda.inflow_data.y_values = &R_VALUES;
    pp_data_fda.inflow_data.z_values = &PHI_VALUES;
    pp_data_fda.inflow_data.array = &VELOCITY_VALUES;

    // calculation of flow rate (use volume-based computation)
    pp_data_fda.mean_velocity_data.calculate = true;
    pp_data_fda.mean_velocity_data.filename_prefix = OUTPUT_FOLDER + FILENAME_FLOWRATE;
    Tensor<1,dim,double> direction; direction[2] = 1.0;
    pp_data_fda.mean_velocity_data.direction = direction;
    pp_data_fda.mean_velocity_data.write_to_file = true;
  }
  else if(domain_id == 2)
  {
    // write output for visualization of results
    pp_data.output_data.write_output = WRITE_OUTPUT;
    pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
    pp_data.output_data.output_name = OUTPUT_NAME_2;
    pp_data.output_data.output_start_time = OUTPUT_START_TIME_NOZZLE;
    pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.mean_velocity.calculate = true;
    pp_data.output_data.mean_velocity.sample_start_time = SAMPLE_START_TIME;
    pp_data.output_data.mean_velocity.sample_end_time = SAMPLE_END_TIME;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.degree = param.degree_u;

    pp_data_fda.pp_data = pp_data;

    // evaluation of quantities along lines
    pp_data_fda.line_plot_data.write_output = true;
    pp_data_fda.line_plot_data.filename_prefix = OUTPUT_FOLDER;
    pp_data_fda.line_plot_data.statistics_data.calculate_statistics = true;
    pp_data_fda.line_plot_data.statistics_data.sample_start_time = SAMPLE_START_TIME;
    pp_data_fda.line_plot_data.statistics_data.sample_end_time = END_TIME;
    pp_data_fda.line_plot_data.statistics_data.sample_every_timesteps = SAMPLE_EVERY_TIMESTEPS;
    pp_data_fda.line_plot_data.statistics_data.write_output_every_timesteps = WRITE_OUTPUT_EVERY_TIMESTEPS;

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
    pp_data_fda.line_plot_data.lines.push_back(axial_profile);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z1);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z2);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z3);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z4);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z5);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z6);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z7);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z8);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z9);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z10);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z11);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z12);
  }
}

// specialization for dim=2, which needs to be implemented explicitly since the
// function above can not be compiled for dim=2.
void
initialize_postprocessor_data(PostProcessorDataFDA<2> pp_data_fda,
                              InputParameters const   &param,
                              unsigned int const      domain_id)
{
  (void)pp_data_fda;
  (void)param;
  (void)domain_id;

  AssertThrow(false, ExcMessage("This test case is only implemented for dim = 3."));
}

// two-domain solver
template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param, unsigned int const domain_id)
{
  std::shared_ptr<PostProcessorBase<dim,Number> > pp;

  PostProcessorDataFDA<dim> pp_data_fda;
  initialize_postprocessor_data(pp_data_fda, param, domain_id);

  pp.reset(new PostProcessorFDA<dim,Number>(pp_data_fda));

  return pp;
}

// standard case of single-domain solver: call function with domain_id = 2
template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param)
{
  std::shared_ptr<PostProcessorBase<dim,Number> > pp;

  PostProcessorDataFDA<dim> pp_data_fda;
  initialize_postprocessor_data(pp_data_fda, param, 2 /* nozzle domain */);

  pp.reset(new PostProcessorFDA<dim,Number>(pp_data_fda));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
