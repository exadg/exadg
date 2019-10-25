/*
 * stokes_curl_flow.h
 *
 *  Created on: Oct 18, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 4;
unsigned int const DEGREE_MAX = 4;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.0e-5;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Steady;
  param.equation_type = EquationType::Stokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.right_hand_side = true;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 1.0e3;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  param.time_step_size = 0.1;
  param.order_time_integrator = 1; // 1; // 2; // 3;
  param.start_with_low_order = true; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/10;
 
  //pseudo-timestepping
  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement;
  param.abs_tol_steady = 1e-8;
  param.rel_tol_steady = 1e-8;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;
  param.adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e1;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  param.type_penalty_parameter = TypePenaltyParameter::ViscousAndConvectiveTerms;
  param.add_penalty_terms_to_monolithic_system = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-8);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  param.preconditioner_viscous = PreconditionerViscous::Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = param.order_time_integrator-1;
  param.rotational_formulation = true;

  // momentum step

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;


  // COUPLED NAVIER-STOKES SOLVER

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
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
  (void)periodic_faces;

  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);
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


template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
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
  pp_data.output_data.output_folder = "output/stokes_curl_flow/vtu/";
  pp_data.output_data.output_name = "stokes_curl_flow";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time); // /10;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.degree = param.degree_u;

  // calculation of velocity error
  pp_data.error_data_u.analytical_solution_available = true;
  pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
  pp_data.error_data_u.calculate_relative_errors = true;
  pp_data.error_data_u.error_calc_start_time = param.start_time;
  pp_data.error_data_u.error_calc_interval_time = (param.end_time - param.start_time);
  pp_data.error_data_u.name = "velocity";

  // ... pressure error
  pp_data.error_data_p.analytical_solution_available = true;
  pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
  pp_data.error_data_p.calculate_relative_errors = true;
  pp_data.error_data_p.error_calc_start_time = param.start_time;
  pp_data.error_data_p.error_calc_interval_time = (param.end_time - param.start_time);
  pp_data.error_data_p.name = "pressure";

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_SHAHBAZI_H_ */
