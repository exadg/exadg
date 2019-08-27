/*
 * poiseuille.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 2;
unsigned int const DEGREE_MAX = 2;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;


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

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = PROBLEM_TYPE;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.use_outflow_bc_convective_term = true;
  param.right_hand_side = periodicBCs; //prescribe body force in x-direction in case of periodic BC's


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 10.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = MAX_VELOCITY;
  param.cfl = 2.0e-1;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2; // 1; // 2; // 3;
  param.start_with_low_order = true; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement; //ResidualSteadyNavierStokes;
  param.abs_tol_steady = 1.e-12;
  param.rel_tol_steady = 1.e-6;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/10;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = periodicBCs;


  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-20,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-20, 1.e-12);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-20,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-14,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-20, 1.e-6, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-10,1.e-6);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES; //GMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev; //Jacobi; //Chebyshev; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
  if(periodicBCs == true)
  {
    std::vector<unsigned int> repetitions({1,1});
    Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
    GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

    //periodicity in x-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
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
    auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 0, periodic_faces);
    triangulation->add_periodicity(periodic_faces);
  }
  else if(symmetryBC == true)
  {
    double y_upper_wall = 0.0;
    std::vector<unsigned int> repetitions({4,1});
    Point<dim> point1(0.0,-H/2.), point2(L,y_upper_wall);
    GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
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
    GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
  }

  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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
          result = 1.0/VISCOSITY * pressure_gradient * (pow(p[1],2.0)-1.0)/2.0 * (t<T ? std::sin(pi/2.*t/T) : 1.0);
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
         result = 0.25;
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

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = "output/poiseuille/vtu/";
  pp_data.output_data.output_name = "test";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/100;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_velocity_magnitude = true;
  pp_data.output_data.write_vorticity_magnitude = true;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.write_q_criterion = true;
  pp_data.output_data.degree = param.degree_u;

  // calculation of error
  if(INFLOW_PROFILE == InflowProfile::ParabolicProfile)
  {
    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.error_calc_start_time = param.start_time;
    pp_data.error_data_u.error_calc_interval_time = (param.end_time - param.start_time)/100;
    pp_data.error_data_u.name = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.error_calc_start_time = param.start_time;
    pp_data.error_data_p.error_calc_interval_time = (param.end_time - param.start_time)/100;
    pp_data.error_data_p.name = "pressure";
  }

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
