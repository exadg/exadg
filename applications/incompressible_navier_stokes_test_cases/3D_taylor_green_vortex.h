/*
 * Beltrami.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_


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
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = 3; //REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = 0; //REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double Re = 1600.0;

const double V_0 = 1.0;
const double L = 1.0;
const double p_0 = 0.0;

const double VISCOSITY = V_0*L/Re;
const double MAX_VELOCITY = V_0;
const double CHARACTERISTIC_TIME = L/V_0;

std::string OUTPUT_FOLDER = "output/taylor_green_vortex/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "Re1600_l3_k32_CFL_0-2";

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 20.0*CHARACTERISTIC_TIME;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDFPressureCorrection; //BDFDualSplittingScheme; //BDFPressureCorrection; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit; //Implicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  max_velocity = MAX_VELOCITY;
  cfl = 0.2;
  cfl_oif = cfl/5.0;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-3; // 1.0e-4;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


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

  // special case: pure DBC's (only periodic BCs -> pure_dirichlet_bc = true)
  pure_dirichlet_bc = true;

  // div-div and continuity penalty
  // these parameters are only used for the coupled solver
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_use_boundary_data = false;
  continuity_penalty_factor = divergence_penalty_factor;

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
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-6;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-20;
  rel_tol_projection = 1.e-12;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // convective step

  // nonlinear solver
  newton_solver_data_convective.abs_tol = 1.e-20;
  newton_solver_data_convective.rel_tol = 1.e-6;
  newton_solver_data_convective.max_iter = 100;
  // linear solver
  abs_tol_linear_convective = 1.e-20;
  rel_tol_linear_convective = 1.e-3;
  max_iter_linear_convective = 1e4;
  use_right_preconditioning_convective = true;
  max_n_tmp_vectors_convective = 100;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-6;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix; //VelocityDiffusion; //InverseMassMatrix; //VelocityConvectionDiffusion;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_momentum_linear = 1.e-12;
  rel_tol_momentum_linear = 1.e-6;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = order_time_integrator-1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;
  scaling_factor_continuity = 1.0e0;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-6;
  max_iter_linear = 1e3;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix; //VelocityDiffusion;
  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-6;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::PressureConvectionDiffusion; //CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING

  // print input parameters
  print_input_parameters = true; //false;

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.write_divergence = false;
  output_data.write_velocity_magnitude = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_q_criterion = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // kinetic energy
  kinetic_energy_data.calculate = true;
  kinetic_energy_data.calculate_every_time_steps = 1;
  kinetic_energy_data.viscosity = VISCOSITY;
  kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  This function is used to prescribe initial conditions for the velocity field
 */
template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity (const unsigned int  n_components = dim,
                           const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~InitialSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double InitialSolutionVelocity<dim>::value(const Point<dim>   &p,
                                           const unsigned int component) const
{
  double result = 0.0;

  if (component == 0)
    result = V_0*std::sin(p[0]/L)*std::cos(p[1]/L)*std::cos(p[2]/L);
  else if (component == 1)
    result = -V_0*std::cos(p[0]/L)*std::sin(p[1]/L)*std::cos(p[2]/L);
  else if (component == 2)
    result = 0.0;

  return result;
}

/*
 *  This function is used to prescribe initial conditions for the pressure field
 */
template<int dim>
class InitialSolutionPressure : public Function<dim>
{
public:
  InitialSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  virtual ~InitialSolutionPressure(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double InitialSolutionPressure<dim>::value(const Point<dim>    &p,
                                           const unsigned int  /* component */) const
{
  double t = this->get_time();
  double result = 0.0;

  result = p_0 + V_0 * V_0 / 16.0 * (std::cos(2.0*p[0]/L) + std::cos(2.0*p[1]/L)) * (std::cos(2.0*p[2]/L) + 2.0);

  return result;
}

/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
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

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSide<dim>::value(const Point<dim>   &p,
                                  const unsigned int component) const
 {
   double result = 0.0;
   return result;
 }


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>              &triangulation,
    unsigned int const                                     n_refine_space,
    std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >                 &periodic_faces)
{
  const double pi = numbers::PI;
  const double left = - pi * L, right = pi * L;
  GridGenerator::hyper_cube(triangulation,left,right);

  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));
  // periodicity in x-,y-, and z-direction
  // x-direction
  triangulation.begin()->face(0)->set_all_boundary_ids(0);
  triangulation.begin()->face(1)->set_all_boundary_ids(1);
  // x-direction
  triangulation.begin()->face(2)->set_all_boundary_ids(2);
  triangulation.begin()->face(3)->set_all_boundary_ids(3);
  // z-direction
  triangulation.begin()->face(4)->set_all_boundary_ids(4);
  triangulation.begin()->face(5)->set_all_boundary_ids(5);

  GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation.add_periodicity(periodic_faces);

  // perform global refinements
  triangulation.refine_global(n_refine_space);

  // test case with pure periodic BC
  // boundary descriptors remain empty for velocity and pressure
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new InitialSolutionPressure<dim>());

  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new ZeroFunction<dim>(1));
}

// Postprocessor
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim,Number> >
construct_postprocessor(InputParametersNavierStokes<dim> const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_ */
