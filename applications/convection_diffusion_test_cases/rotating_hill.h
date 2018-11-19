/*
 * rotating_hill.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>


/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 6;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = 3;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;


void ConvDiff::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::Convection;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  diffusivity = 0.0;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK; //BDF; //ExplRK;

  // Explicit RK
  time_integrator_rk = TimeIntegratorRK::ExplRK3Stage7Reg2;

  // BDF
  order_time_integrator = 2; // instabilities for BDF 3 and 4
  start_with_low_order = false;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //ExplicitOIF;
  time_integrator_oif = TimeIntegratorRK::ExplRK4Stage8Reg2; //ExplRK3Stage7Reg2; //ExplRK4Stage8Reg2;

  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = false;
  time_step_size = 1.e-2;
  cfl_oif = 0.2;
  cfl_number = cfl_oif * 1.0;
  diffusion_number = 0.01;
  exponent_fe_degree_convection = 2.0;
  exponent_fe_degree_diffusion = 3.0;
  c_eff = 1.0e0;

  // SPATIAL DISCRETIZATION
  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::GMRES;
  abs_tol = 1.e-20;
  rel_tol = 1.e-8;
  max_iter = 1e3;
  max_n_tmp_vectors = 100;
  preconditioner = Preconditioner::BlockJacobi; //None; //InverseMassMatrix; //PointJacobi; //BlockJacobi; //Multigrid;
  update_preconditioner = true;

  // BlockJacobi (these parameters are also relevant if used as a smoother in multigrid)
  implement_block_diagonal_preconditioner_matrix_free = true;
  preconditioner_block_diagonal = PreconditionerBlockDiagonal::InverseMassMatrix;
  block_jacobi_solver_data = SolverData(1000,1.e-12,1.e-2);

  // Multigrid
  multigrid_data.type = MultigridType::hMG;
  mg_operator_type = MultigridOperatorType::ReactionConvection;

  // MG smoother
  multigrid_data.smoother = MultigridSmoother::Jacobi; //GMRES; //Chebyshev; //ChebyshevNonsymmetricOperator;

  // MG smoother data
  multigrid_data.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi;
  multigrid_data.gmres_smoother_data.number_of_iterations = 5;

  // MG smoother data: Jacobi smoother
  multigrid_data.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //None; //PointJacobi; //BlockJacobi;
  multigrid_data.jacobi_smoother_data.number_of_smoothing_steps = 5;
  multigrid_data.jacobi_smoother_data.damping_factor = 0.8;

  // MG coarse grid solver
  multigrid_data.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner; //GMRES_NoPreconditioner; //Chebyshev; //GMRES_Jacobi;



  // NUMERICAL PARAMETERS
  use_cell_based_face_loops = true;
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  output_data.write_output = true;
  output_data.output_folder = "output_conv_diff/";
  output_data.output_name = "rotating_hill";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.number_of_patches = FE_DEGREE;

  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  output_solver_info_every_timesteps = 1e6;

  // restart
  restart_data.write_restart = true;
  restart_data.filename = "output_conv_diff/rotating_hill";
  restart_data.interval_time = 0.4;
}


/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solution
 */

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution (const unsigned int  n_components = 1,
            const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~Solution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim>    &p,
                                      const unsigned int  /* component */) const
{
  double t = this->get_time();

  double radius = 0.5;
  double omega = 2.0*numbers::PI;
  double center_x = -radius*std::sin(omega*t);
  double center_y = +radius*std::cos(omega*t);
  double result = std::exp(-50*pow(p[0]-center_x,2.0)-50*pow(p[1]-center_y,2.0));

  return result;
}

/*
 *  Right-hand side
 */

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide (const unsigned int   n_components = 1,
                 const double         time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~RightHandSide(){};

  virtual double value (const Point<dim>    &p,
                       const unsigned int  component = 0) const;
};

template<int dim>
double RightHandSide<dim>::value(const Point<dim>     &/*p*/,
                                const unsigned int   /* component */) const
{
  double result = 0.0;
  return result;
}

/*
 *  Neumann boundary condition
 */

template<int dim>
class NeumannBoundary : public Function<dim>
{
public:
  NeumannBoundary (const unsigned int n_components = 1,
                   const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~NeumannBoundary(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double NeumannBoundary<dim>::value(const Point<dim>   &p,
                                   const unsigned int /* component */) const
{
  double result = 0.0;
  return result;
}

/*
 *  Velocity field
 */

template<int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField (const unsigned int n_components = dim,
                 const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~VelocityField(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double VelocityField<dim>::value(const Point<dim>   &point,
                                 const unsigned int component) const
{
  double value = 0.0;

  if(component == 0)
    value = -point[1]*2.0*numbers::PI;
  else if(component ==1)
    value =  point[0]*2.0*numbers::PI;

  // TODO time dependent velocity field to test adaptive time stepping
//  double t = this->get_time();
//  value *= 1.0-t;

  return value;
}

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>           &triangulation,
    unsigned int const                                  n_refine_space,
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  triangulation.refine_global(n_refine_space);

  std::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new Solution<dim>());
  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,analytical_solution));
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new Solution<dim>());

  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  std::shared_ptr<Function<dim> > velocity;
  velocity.reset(new VelocityField<dim>());

  field_functions->analytical_solution = analytical_solution;
  field_functions->right_hand_side = right_hand_side;
  field_functions->velocity = velocity;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>(1));
}


#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_ */
