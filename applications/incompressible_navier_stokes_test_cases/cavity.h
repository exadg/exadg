/*
 * cavity.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

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
const ProblemType PROBLEM_TYPE = ProblemType::Steady;
const double L = 1.0;

std::string OUTPUT_FOLDER = "output/cavity/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = PROBLEM_TYPE;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = false;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 10.0;
  param.viscosity = 1.0e-2;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Steady;
  param.temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  param.adaptive_time_stepping = false;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.max_velocity = 1.0;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  // Explicit: CFL_crit = 0.35 (0.4 unstable), ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable)
  param.cfl_oif = 3.0;
  param.cfl = param.cfl_oif * 1.0;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.dt_refinements = REFINE_TIME_MIN;

  // pseudo-timestepping for steady-state problems
  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  param.abs_tol_steady = 1.e-12;
  param.rel_tol_steady = 1.e-8;

  // restart
  param.restart_data.write_restart = false;
  param.restart_data.interval_time = 5.0;
  param.restart_data.interval_wall_time = 1.e6;
  param.restart_data.interval_time_steps = 1e8;
  param.restart_data.filename = "output/cavity/cavity_restart";

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time_steps = 1;
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
  param.pure_dirichlet_bc = true;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.add_penalty_terms_to_monolithic_system = true;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;
  param.solver_data_block_diagonal = SolverData(1000, 1.e-12, 1.e-2, 1000);
  param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8,100);
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
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES; //FGMRES;
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  else
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = true;
  param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  param.multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_momentum.smoother_data.iterations = 5;
  param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
  param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES; //FGMRES;
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 1000);
  else
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 1000);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;

  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  param.exact_inversion_of_velocity_block = false; // true;
  param.solver_data_velocity_block = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  param.multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.exact_inversion_of_laplace_operator = false;
  param.solver_data_pressure_block = SolverData(1e4, 1.e-12, 1.e-6, 100);
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
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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

namespace IncNS
{

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
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
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/100;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_streamfunction = true; //false;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.degree = param.degree_u;

  // line plot data
  pp_data.line_plot_data.write_output = false;
  pp_data.line_plot_data.filename_prefix = OUTPUT_FOLDER;

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
  pp_data.line_plot_data.lines.push_back(vert_line);

  // horizontal line
  hor_line.begin = Point<dim>(0.0,0.5);
  hor_line.end = Point<dim>(1.0,0.5);
  hor_line.name = "hor_line";
  hor_line.n_points = 10001; //2001;
  hor_line.quantities.push_back(quantity_u);
  //hor_line.quantities.push_back(quantity_p);
  pp_data.line_plot_data.lines.push_back(hor_line);

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
