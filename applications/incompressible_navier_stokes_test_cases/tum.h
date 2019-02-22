/*
 * tum.h
 *
 *  Created on: Sep 8, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_


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
const ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
const double L = 1.0;
const double MAX_VELOCITY = 1.0;
const double VISCOSITY = 1.0e-4; //TODO //1.0e-3;

std::string OUTPUT_FOLDER = "output/tum/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test"; //"Re1e3_l6_k21_outflow_bc_dual_splitting";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE; // PROBLEM_TYPE is also needed somewhere else
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  use_outflow_bc_convective_term = true;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 50.0;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  // best practice: use adaptive time stepping for this test case to avoid adjusting the CFL number
  adaptive_time_stepping = true;
  max_velocity = 1.0;
  // typical CFL values for BDF2 (constant time step size), exponent = 1.5, k_u = 2, MAX_VELOCITY = 1.0
  // Re = 1e3: cfl = 0.025
  // Re = 1e4: cfl = 0.0125 and 0.003125 (l=5)
  cfl = 0.25; //TODO // 0.1; //0.025;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 5.0e-2;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;

  // pseudo-timestepping for steady-state problems
  convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  abs_tol_steady = 1.e-12;
  rel_tol_steady = 1.e-10;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES; //GMRES; //FGMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = true;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::FGMRES; //FGMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 100);

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
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/200;
  output_data.write_divergence = true;
  output_data.write_streamfunction = false;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  output_solver_info_every_timesteps = 1e2; //1e3;
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
      if(component == 0 && (std::abs(p[0])<1.0e-15))
        result = MAX_VELOCITY;
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      const double T = 0.5;
      const double pi = numbers::PI;
      if(component == 0 && (std::abs(p[0])<1.0e-15))
      {
        result = t<T ? std::sin(pi/2.*t/T) : 1.0;

        result *= MAX_VELOCITY; //*(1.0 - std::pow(p[1]/L+0.5,2.0));
      }
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
    Triangulation<dim> tria_h1, tria_h2, tria_h3, tria_v1, tria_v2, tria_v3, tria_v4, tria_v5;

    GridGenerator::subdivided_hyper_rectangle(tria_h1,
                                              std::vector<unsigned int>({4,1}),
                                              Point<dim>(0.0,0.0),
                                              Point<dim>(4.0*L,-1.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_h2,
                                              std::vector<unsigned int>({1,1}),
                                              Point<dim>(4.0*L,-4.0*L),
                                              Point<dim>(5.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_h3,
                                              std::vector<unsigned int>({5,1}),
                                              Point<dim>(5.0*L,0.0*L),
                                              Point<dim>(10.0*L,-1.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v1,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(1.0*L,-1.0*L),
                                              Point<dim>(2.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v2,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(3.0*L,-1.0*L),
                                              Point<dim>(4.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v3,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(5.0*L,-1.0*L),
                                              Point<dim>(6.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v4,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(7.0*L,-1.0*L),
                                              Point<dim>(8.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v5,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(9.0*L,-1.0*L),
                                              Point<dim>(10.0*L,-5.0*L));

    // merge
    Triangulation<dim> tmp1, tmp2;
    GridGenerator::merge_triangulations (tria_h1, tria_v1, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v2, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_h2, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v3, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_h3, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v4, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_v5, *triangulation);

    // global refinements
    triangulation->refine_global(n_refine_space);
  }
  else if(dim == 3)
  {
    AssertThrow(false, ExcMessage("NotImplemented"));
  }

  // set boundary indicator
  // all boundaries have ID = 0 by default -> Dirichlet boundaries
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // inflow boundary
     if ((std::fabs(cell->face(face_number)->center()(0))< 1e-12))
        cell->face(face_number)->set_boundary_id (1);

     // outflow boundary
     if ((std::fabs(cell->face(face_number)->center()(1) - (-5.0*L))< 1e-12) && (cell->face(face_number)->center()(0)- 9.0*L)>=0)
        cell->face(face_number)->set_boundary_id (2);
    }
  }

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new AnalyticalSolutionVelocity<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(2,new Functions::ZeroFunction<dim>(1)));
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



#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_ */
