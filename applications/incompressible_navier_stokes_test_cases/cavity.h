/*
 * Cavity.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

namespace IncNS
{

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 2;
unsigned int const REFINE_STEPS_SPACE_MAX = 2; //REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const ProblemType PROBLEM_TYPE = ProblemType::Steady;
const double L = 1.0;

std::string OUTPUT_FOLDER = "output/cavity/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE; // PROBLEM_TYPE is also needed somewhere else
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 10.0;
  viscosity = 1.0e-1;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  adaptive_time_stepping = false;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = 1.0;
  cfl_exponent_fe_degree_velocity = 1.5;
  // Explicit: CFL_crit = 0.35 (0.4 unstable), ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable)
  cfl_oif = 3.0;
  cfl = cfl_oif * 1.0;
  time_step_size = 1.0e-1;
  order_time_integrator = 2;
  start_with_low_order = true;

  // pseudo-timestepping for steady-state problems
  convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  abs_tol_steady = 1.e-12;
  rel_tol_steady = 1.e-8;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;

  // div-div and continuity penalty
  use_divergence_penalty = false;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = false;
  continuity_penalty_factor = divergence_penalty_factor;
  add_penalty_terms_to_monolithic_system = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-8);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES; //FGMRES;
  if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  else
    solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = true;
  multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data_momentum.smoother_data.iterations = 5;
  multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
  multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-10);

  // linear solver
  solver_coupled = SolverCoupled::FGMRES; //FGMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 1000);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionConvectionDiffusion;
  multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::GMRES;
  multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data_velocity_block.smoother_data.iterations = 5;
  multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;

  multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  exact_inversion_of_velocity_block = false; // true;
  solver_data_velocity_block = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  exact_inversion_of_laplace_operator = false;
  solver_data_pressure_block = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/100;
  output_data.write_divergence = true;
  output_data.write_streamfunction = false;
  output_data.write_processor_id = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/10;

  // line plot data
  line_plot_data.write_output = false;
  line_plot_data.filename_prefix = OUTPUT_FOLDER;

  // which quantities
  Quantity* quantity_u = new Quantity();
  quantity_u->type = QuantityType::Velocity;
//  Quantity quantity_p;
//  quantity_p.type = QuantityType::Pressure;

  // lines
  Line<dim> vert_line, hor_line;

  // vertical line
  vert_line.begin = Point<dim>(0.5,0.0);
  vert_line.end = Point<dim>(0.5,1.0);
  vert_line.name = "vert_line";
  vert_line.n_points = 100001; //2001;
  vert_line.quantities.push_back(quantity_u);
  //vert_line.quantities.push_back(quantity_p);
  line_plot_data.lines.push_back(vert_line);

  // horizontal line
  hor_line.begin = Point<dim>(0.0,0.5);
  hor_line.end = Point<dim>(1.0,0.5);
  hor_line.name = "hor_line";
  hor_line.n_points = 10001; //2001;
  hor_line.quantities.push_back(quantity_u);
  //hor_line.quantities.push_back(quantity_p);
  line_plot_data.lines.push_back(hor_line);

  // restart
  restart_data.write_restart = true;
  restart_data.interval_time = 5.0;
  restart_data.interval_wall_time = 1.e6;
  restart_data.interval_time_steps = 1e8;
  restart_data.filename = "output/cavity/cavity_restart";
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
    double t = this->get_time();
    double result = 0.0;

    // constant velocity
    if(PROBLEM_TYPE == ProblemType::Steady)
    {
      if(component == 0 && (std::abs(p[1]-L)<1.0e-15))
        result = 1.0;
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      const double T = 0.5;
      const double pi = numbers::PI;
      if(component == 0 && (std::abs(p[1]-L)<1.0e-15))
        result = t<T ? std::sin(pi/2.*t/T) : 1.0;
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
  if(dim == 2)
  {
    Point<dim> point1(0.0,0.0), point2(L,L);
    GridGenerator::hyper_rectangle(*triangulation,point1,point2);
    triangulation->refine_global(n_refine_space);
  }
  else if(dim == 3)
  {
    const double left = 0.0, right = L;
    GridGenerator::hyper_cube(*triangulation,left,right);
    triangulation->refine_global(n_refine_space);
  }

  // all boundaries have ID = 0 by default -> Dirichlet boundaries

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
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
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
