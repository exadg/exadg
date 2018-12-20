/*
 * Poiseuille.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_

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
const double MAX_VELOCITY = 1.0;
const double VISCOSITY = 1.0e-1;

const double H = 2.0;
const double L = 4.0;

bool periodicBCs = false;

bool symmetryBC = false;

enum class InflowProfile { ConstantProfile, ParabolicProfile };
const InflowProfile INFLOW_PROFILE = InflowProfile::ParabolicProfile;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE; // PROBLEM_TYPE is also needed somewhere else
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  use_outflow_bc_convective_term = true;
  right_hand_side = periodicBCs; //prescribe body force in x-direction in case of perodic BC's


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 10.0;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = MAX_VELOCITY; // MAX_VELOCITY is also needed somewhere else
  cfl = 2.0e-1;
  time_step_size = 1.0e-1;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;

  convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement; //ResidualSteadyNavierStokes;
  abs_tol_steady = 1.e-12;
  rel_tol_steady = 1.e-6;

  // SPATIAL DISCRETIZATION

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = periodicBCs;


  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-20,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

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
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-14,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-20, 1.e-6, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-10,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::FGMRES; //GMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  multigrid_data_velocity_block.smoother = MultigridSmoother::Chebyshev; //Jacobi; //Chebyshev; //GMRES;

  // GMRES smoother data
  multigrid_data_velocity_block.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_velocity_block.gmres_smoother_data.number_of_iterations = 5;

  // Jacobi smoother data
  multigrid_data_velocity_block.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_velocity_block.jacobi_smoother_data.number_of_smoothing_steps = 5;
  multigrid_data_velocity_block.jacobi_smoother_data.damping_factor = 0.7;

  multigrid_data_velocity_block.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner; //NoPreconditioner; //Chebyshev; //Chebyshev; //ChebyshevNonsymmetricOperator;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  multigrid_data_pressure_block.chebyshev_smoother_data.smoother_poly_degree = 5;
  multigrid_data_pressure_block.coarse_solver = MultigridCoarseGridSolver::Chebyshev;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = "output/poiseuille/vtu/";
  output_data.output_name = "poiseuille";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/100;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.write_velocity_magnitude = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_processor_id = true;
  output_data.write_q_criterion = true;
  output_data.mean_velocity.calculate = false;
  output_data.mean_velocity.sample_start_time = start_time;
  output_data.mean_velocity.sample_end_time = end_time;
  output_data.mean_velocity.sample_every_timesteps = 1;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
    error_data.analytical_solution_available = true;
  else
    error_data.analytical_solution_available = false;

  error_data.calculate_relative_errors = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;
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

    // initial velocity field = 0

    //BC's specified below only relevant if periodicBCs == false
    if(PROBLEM_TYPE == ProblemType::Steady)
    {
      if(INFLOW_PROFILE == InflowProfile::ConstantProfile)
      {
        if(component == 0 && (std::abs(p[0])<1.0e-12))
          result = MAX_VELOCITY;
      }
      else if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
      {
        const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
        if(component == 0)
          result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0;
      }
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      const double pi = numbers::PI;
      double T = 1.0e0;

      if(INFLOW_PROFILE == InflowProfile::ConstantProfile)
      {
        // ensure that the function is only "active" at the left boundary and if component == 0
        if(component == 0 && (std::abs(p[0])<1.0e-12))
          result = MAX_VELOCITY * (t<T ? std::sin(pi/2.*t/T) : 1.0);
      }
      else if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
      {
        const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
        if(component == 0)
        {
  //        result = 1.0/VISCOSITY * pressure_gradient * (pow(p[1],2.0)-1.0)/2.0 * (t<T ? std::sin(pi/2.*t/T) : 1.0);

          result = 1.0/VISCOSITY * pressure_gradient * (pow(p[1],2.0)-1.0)/2.0 * std::sin(pi*t/T);
        }
      }
    }

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
    double t = this->get_time();
    double result = 0.0;

    if(PROBLEM_TYPE == ProblemType::Steady)
    {
      if(INFLOW_PROFILE == InflowProfile::ConstantProfile)
      {
        // For this inflow profile no analytical solution is available.
        // Set the pressure to zero at the outflow boundary. This is
        // already done since result is initialized with a value of 0.0.
      }
      else if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
      {
        // pressure decreases linearly in flow direction
        const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
        result = (p[0]-L)*pressure_gradient;
      }
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      if(INFLOW_PROFILE == InflowProfile::ConstantProfile)
      {
        // For this inflow profile no analytical solution is available.
        // Set the pressure to zero at the outflow boundary. This is
        // already done since result is initialized with a value of 0.0.
      }
      else if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
      {
        // parabolic velocity profile
        const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
        const double pi = numbers::PI;
        double T = 1.0e0;
        // note that this is the steady state solution that would correspond to a
        // steady velocity field at time t
        result = (p[0]-L) * pressure_gradient * (t<T ? std::sin(pi/2.*t/T) : 1.0);
      }
    }
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
    (void)p;
    (void)component;

    double result = 0.0;

    // The Neumann velocity boundary condition that is consistent with the analytical solution
    // (in case of a parabolic inflow profile) is (grad U)*n = 0.

    // Hence:
    // If the viscous term is written in Laplace formulation, prescribe result = 0 as Neumann BC
    // If the viscous term is written in Divergence formulation, the following boundary condition
    // has to be used to ensure that (grad U)*n = 0:
    // (grad U + (grad U)^T)*n = (grad U)^T * n

  //  if(component==1)
  //    result = - MAX_VELOCITY * 2.0 * p[1];

    return result;
  }
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  /*component*/) const
  {
    // do nothing (result = 0) since we are interested in a steady state solution
    double result = 0.0;
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

     if(periodicBCs == true)
     {
     if(component==0)
       result = 0.02;
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
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  if(periodicBCs == true)
  {
    std::vector<unsigned int> repetitions({1,1});
    Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    //periodicity in x-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
           cell->face(face_number)->set_boundary_id (0+10);
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1+10);
      }
    }
    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
  }
  else if(symmetryBC == true)
  {
    double y_upper_wall = 0.0;
    std::vector<unsigned int> repetitions({4,1});
    Point<dim> point1(0.0,-H/2.), point2(L,y_upper_wall);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);

       // upper wall symmetry BC
       if ((std::fabs(cell->face(face_number)->center()(1) - y_upper_wall)< 1e-12))
          cell->face(face_number)->set_boundary_id (2);
      }
    }
  }
  else // inflow at left boundary, no-slip on upper and lower wall, outflow at right boundary
  {
    std::vector<unsigned int> repetitions({2,1});
    Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
  }

  triangulation.refine_global(n_refine_space);

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new NeumannBoundaryVelocity<dim>()));

  if(symmetryBC == true)
  {
    // slip boundary condition: always u*n=0
    // function will not be used -> use ZeroFunction
    boundary_descriptor_velocity->symmetry_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));
  }

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new PressureBC_dudt<dim>()));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new AnalyticalSolutionPressure<dim>()));

  if(symmetryBC == true)
  {
    // On symmetry boundaries, a Neumann BC is prescribed for the pressure.
    // -> prescribe dudt for dual-splitting scheme, which is equal to zero since
    // (du/dt)*n = d(u*n)/dt = d(0)/dt = 0, i.e., the time derivative term is multiplied by the normal vector
    // and the normal velocity is zero (= symmetry boundary condition).
    boundary_descriptor_pressure->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));
  }
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
//  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
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
  pp_data.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
