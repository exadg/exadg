/*
 * Kovasznay.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_

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
unsigned int const FE_DEGREE_VELOCITY = 2;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::LaplaceFormulation;
const double VISCOSITY = 2.5e-2;
enum class InitializeSolutionWith { ZeroFunction, AnalyticalSolution};
const InitializeSolutionWith INITIALIZE_SOLUTION_WITH = InitializeSolutionWith::AnalyticalSolution;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FORMULATION_VISCOUS_TERM;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = 3.6;
  cfl = 1.0e-2;
  time_step_size = 1.0e-3;
  order_time_integrator = 3; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-20,1.e-6,100);
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
  solver_data_viscous = SolverData(1000,1.e-20,1.e-6);
  preconditioner_viscous = PreconditionerViscous::Multigrid;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-20, 1.e-4, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;

  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_coupled = SolverCoupled::FGMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 1000);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Jacobi; //Chebyshev; //GMRES;
  multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_velocity_block.smoother_data.iterations = 5;
  multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = "output/kovasznay/";
  output_data.output_name = "kovasznay";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.write_divergence = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/10;
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
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
       cell->face(face_number)->set_boundary_id (1);
    }
  }

  triangulation->refine_global(n_refine_space);
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

namespace IncNS
{

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

    const double pi = numbers::PI;
    const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(pi,2.0),0.5);
    if (component == 0)
      result = 1.0 - std::exp(lambda*p[0])*std::cos(2*pi*p[1]);
    else if (component == 1)
      result = lambda/2.0/pi*std::exp(lambda*p[0])*std::sin(2*pi*p[1]);

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

    const double pi = numbers::PI;
    const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(pi,2.0),0.5);
    result = 0.5*(1.0-std::exp(2.0*lambda*p[0]));

    return result;
  }
};

template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim> &p,
                const unsigned int component = 0) const
  {
    double result = 0.0;

    if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation)
    {
      const double pi = numbers::PI;
      const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
      if (component == 0)
        result = -lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
      else if (component == 1)
        result = std::pow(lambda,2.0)/2/pi*std::exp(lambda)*std::sin(2*pi*p[1]);
    }
    else if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation)
    {
      const double pi = numbers::PI;
      const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
      if (component == 0)
        result = -2.0*lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
      else if (component == 1)
        result = (std::pow(lambda,2.0)/2/pi+2.0*pi)*std::exp(lambda)*std::sin(2*pi*p[1]);
    }
    else
    {
      AssertThrow(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation ||
                  FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }
};

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new NeumannBoundaryVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new AnalyticalSolutionPressure<dim>()));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  if(INITIALIZE_SOLUTION_WITH == InitializeSolutionWith::ZeroFunction)
  {
    initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  }
  else if (INITIALIZE_SOLUTION_WITH == InitializeSolutionWith::AnalyticalSolution)
  {
    initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  }

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
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

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_ */
