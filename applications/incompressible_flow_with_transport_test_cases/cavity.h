/*
 * cavity.h
 *
 *  Created on: Nov 26, 2018
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

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;
unsigned int const FE_DEGREE_SCALAR = FE_DEGREE_VELOCITY;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 4;
unsigned int const REFINE_STEPS_SPACE_MAX = 4;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

// set problem specific parameters like physical dimensions, etc.
double const L = 1.0; // Length of cavity

double const START_TIME = 0.0;
double const END_TIME = 10.0;

// Explicit: CFL_crit = 0.35-0.4 unstable (DivergenceFormulation, upwind_factor=0.5) for BDF2 with constant time steps
//           CFL_crit = 0.32-0.33 (DivergenceFormulation, upwind_factor=0.5!), 0.5-0.6 (ConvectiveFormulation) for BDF2 with adaptive time stepping
// ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable) for ExplRK3Stage7Reg2
double const CFL_OIF = 0.3; //0.32;
double const CFL = CFL_OIF;
double const MAX_VELOCITY = 1.0;
bool const ADAPTIVE_TIME_STEPPING = true;

// output
bool const WRITE_OUTPUT = true;
std::string const OUTPUT_FOLDER = "output_flow_with_transport/cavity/";
std::string const OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "test";
double const OUTPUT_START_TIME = START_TIME;
double const OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/100.0;

// solver info
unsigned int const OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 100;

// restart
bool const WRITE_RESTART = false;
double const RESTART_INTERVAL_TIME = 10.0;

template<int dim>
void IncNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = 1.0e-5;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = MAX_VELOCITY;
  cfl_exponent_fe_degree_velocity = 1.5;
  cfl_oif = CFL_OIF;
  cfl = CFL;
  time_step_size = 1.0e-1;
  order_time_integrator = 2;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // polynomial degrees
  degree_u = FE_DEGREE_VELOCITY;
  degree_p = FE_DEGREE_PRESSURE;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;


  // NUMERICAL PARAMETERS
  implement_block_diagonal_preconditioner_matrix_free = true;
  use_cell_based_face_loops = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulation
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  // use FGMRES for matrix-free BlockJacobi or Multigrid with Krylov methods as smoother/coarse grid solver
  solver_momentum = SolverMomentum::FGMRES;
  solver_data_momentum = SolverData(1e4,1.e-12,1.e-6,100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix; //BlockJacobi; //Multigrid;
  multigrid_operator_type_momentum = MultigridOperatorType::ReactionConvectionDiffusion;
  update_preconditioner_momentum = true;
  multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data_momentum.smoother_data.iterations = 5;
  multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
  multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
  multigrid_data_momentum.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::None;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::GMRES; //FGMRES;
  solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

  // preconditioner linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = WRITE_OUTPUT;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME + "_fluid";
  output_data.output_start_time = OUTPUT_START_TIME;
  output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  output_data.write_processor_id = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // restart
  restart_data.write_restart = WRITE_RESTART;
  restart_data.interval_time = RESTART_INTERVAL_TIME;
  restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_fluid";
}

void ConvDiff::InputParameters::set_input_parameters(unsigned int scalar_index)
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::ConvectionDiffusion;
  type_velocity_field = TypeVelocityField::Numerical;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  if(scalar_index == 0)
  {
    diffusivity = 1.e-5;
  }
  else
  {
    diffusivity = 1.e-3;
  }

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDF;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  order_time_integrator = 2;
  time_integrator_oif = TimeIntegratorRK::ExplRK3Stage7Reg2;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  time_step_size = 1.0e-2;
  cfl_oif = CFL_OIF;
  cfl = CFL;
  max_velocity = MAX_VELOCITY;
  exponent_fe_degree_convection = 1.5;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // polynomial degree
  degree = FE_DEGREE_SCALAR;
  degree_mapping = 1;

  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::GMRES;
  solver_data = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner = Preconditioner::InverseMassMatrix; //BlockJacobi; //Multigrid;
  implement_block_diagonal_preconditioner_matrix_free = true;
  use_cell_based_face_loops = true;
  update_preconditioner = true;

  multigrid_data.type = MultigridType::hMG;
  mg_operator_type = MultigridOperatorType::ReactionConvectionDiffusion;
  // MG smoother
  multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;
  // MG smoother data
  multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data.smoother_data.iterations = 5;

  // MG coarse grid solver
  multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  output_data.write_output = WRITE_OUTPUT;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
  output_data.output_start_time = OUTPUT_START_TIME;
  output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  output_data.degree = FE_DEGREE_SCALAR;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // restart
  restart_data.write_restart = WRITE_RESTART;
  restart_data.interval_time = RESTART_INTERVAL_TIME;
  restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
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
  const double left = 0.0, right = L;
  GridGenerator::hyper_cube(*triangulation,left,right);
  triangulation->refine_global(n_refine_space);

  // set boundary IDs: 0 by default, set upper boundary to 1
  typename Triangulation<dim>::cell_iterator cell;
  for(cell = triangulation->begin(); cell != triangulation->end(); ++cell)
  {
    for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if ((std::fabs(cell->face(face_number)->center()(1) - L)< 1e-12))
      {
         cell->face(face_number)->set_boundary_id(1);
      }
    }
  }
}

/**************************************************************************************/
/*                                                                                    */
/*          FUNCTIONS (ANALYTICAL/INITIAL SOLUTION, BOUNDARY CONDITIONS, etc.)        */
/*                                                                                    */
/**************************************************************************************/

namespace IncNS
{

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC (const unsigned int  n_components = dim,
               const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = 0.0;
    (void)p;

    if(component == 0)
    {
      // Variation of boundary condition: avoid velocity jumps in corners (reduces Newton iterations considerably for convection-dominated problems)
//      result = 4./(L*L) * (-(p[0]-L/2.0)*(p[0]-L/2.0) + L*L/4.0);

      result = 1.0;
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
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new DirichletBC<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
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

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
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

  std::shared_ptr<PostProcessor<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"

namespace ConvDiff
{

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC (const unsigned int n_components = 1,
               const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  /*component = 0*/) const
  {
    return 1.0;
  }
};

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor->dirichlet_bc.insert(pair(1,new DirichletBC<dim>()));
}

template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  field_functions->analytical_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  analytical_solution->solution.reset(new Functions::ZeroFunction<dim>(1));
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
