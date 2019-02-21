/*
 * stokes_curl_flow.h
 *
 *  Created on: Oct 18, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_

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
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 4;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 2;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.0e-5;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Steady;
  equation_type = EquationType::Stokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation; //LaplaceFormulation; //DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0e3;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFPressureCorrection; //BDFCoupledSolution; //BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  max_velocity = 1.0;
  cfl = 2.0e-1;
  time_step_size = 0.1;
  order_time_integrator = 1; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;

  //pseudo-timestepping
  convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement; //ResidualSteadyNavierStokes;
  abs_tol_steady = 1e-8;
  rel_tol_steady = 1e-8;
 
  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;
  adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e1;
  use_continuity_penalty = true;
  continuity_penalty_factor = divergence_penalty_factor;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  type_penalty_parameter = TypePenaltyParameter::ViscousAndConvectiveTerms;
  add_penalty_terms_to_monolithic_system = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-20, 1.e-12);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  preconditioner_viscous = PreconditionerViscous::Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = order_time_integrator-1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)

  // linear solver
  solver_coupled = SolverCoupled::GMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-10, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::InverseMassMatrix; //CahouetChabard; //InverseMassMatrix;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = "output/stokes_curl_flow/vtu/";
  output_data.output_name = "stokes_curl_flow";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time); // /10;
  output_data.write_divergence = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e1;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = 0.0;

    double x = p[0];
    double y = p[1];
    if (component == 0)
      result = x*x*(1-x)*(1-x)*(2*y*(1-y)*(1-y) - 2*y*y*(1-y));
    else if (component == 1)
      result = -1*y*y*(1-y)*(1-y)*(2*x*(1-x)*(1-x) - 2*x*x*(1-x));

    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double result = 0.0;

    result = std::pow(p[0],5.0) + std::pow(p[1],5.0) - 1.0/3.0;

    return result;
  }
};

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

   double value (const Point<dim>    &p,
                 const unsigned int  component = 0) const
   {
     double nu = VISCOSITY;
     double x = p[0];
     double y = p[1];
     double result = 0.0;

     if(component == 0)
     {
       result = -nu * (+ 4 * (1 - x) * (1-x) * y * (1 - y) * (1-y)
                       - 16 * x * (1 - x) * y * (1 - y) * (1-y)
                       + 4 * x * x * y * (1 - y) * (1-y)
                       - 4 * (1 - x) * (1-x) * y * y  * (1 - y)
                       + 16 * x * (1 - x) * y * y  * (1 - y)
                       - 4 * x * x * y * y * (1 - y)
                       - 12 * x * x * (1 - x) * (1 - x) * (1 - y)
                       + 12 * x * x  * (1 - x) * (1-x) * y)
                 + 5 * x * x * x * x;
     }

     if(component == 1)
     {
       result = -nu * (12 * (1 - x) * y * y * (1 - y) * (1-y)
                       - 12 * x * y * y * (1 - y) * (1-y)
                       - 4 * x * (1 - x) * (1-x) * (1 - y) * (1-y)
                       + 16 * x * (1 - x) * (1-x) * y * (1 - y)
                       - 4 * x * (1 - x) * (1-x) * y * y
                       + 4 * x * x * (1 - x) * (1 - y) * (1-y)
                       - 16 * x * x * (1 - x) * y * (1 - y)
                       + 4 * x * x  * (1 - x) * y * y)
                 + 5 * y * y * y * y;
     }

     return result;
   }
 };


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);
  triangulation->refine_global(n_refine_space);

  // test case with pure Dirichlet boundary conditions for velocity
  // all boundaries have ID = 0 by default

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

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

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_ */
