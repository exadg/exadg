/*
 * ConstWindConstRHS.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_CONSTWINDCONSTRHS_H_
#define APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_CONSTWINDCONSTRHS_H_





/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// constant source term inside rectangular domain
// pure dirichlet boundary conditions (homogeneous)
// use constant advection velocity -> boundary layer

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 1;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 1;
const unsigned int REFINE_STEPS_SPACE_MAX = 8;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

// problem specific parameters
const double START_TIME = 0.0;
const double DIFFUSIVITY = 1.0e0;

void InputParametersConvDiff::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Steady;
  equation_type = EquationType::ConvectionDiffusion;
  right_hand_side = true;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = 1.0;
  diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDF;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  order_time_integrator = 2;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepUserSpecified;
  time_step_size = 1.0e-1;
  cfl_number = 0.2;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION
  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::GMRES;
  abs_tol = 1.e-20;
  rel_tol = 1.e-8;
  max_iter = 1e4;
  max_n_tmp_vectors = 30;
  preconditioner = Preconditioner::BlockJacobi; //MultigridConvectionDiffusion; //MultigridDiffusion; //MultigridConvectionDiffusion;
  // MG smoother
  multigrid_data.smoother = MultigridSmoother::GMRES; //GMRES; //Chebyshev; //ChebyshevNonsymmetricOperator;
  // MG smoother data
  multigrid_data.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::PointJacobi;
  multigrid_data.gmres_smoother_data.number_of_iterations = 5;
  // MG coarse grid solver
  multigrid_data.coarse_solver = MultigridCoarseGridSolver::GMRES_Jacobi; //GMRES_NoPreconditioner; //Chebyshev; //GMRES_Jacobi;

  update_preconditioner = false;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = false;
  output_data.write_output = false;
  output_data.output_prefix = "const_wind_const_rhs";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time);// /20;
  output_data.number_of_patches = FE_DEGREE;

  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  output_solver_info_every_timesteps = 1e2;
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
class AnalyticalSolution : public Function<dim>
{
public:
  AnalyticalSolution (const unsigned int  n_components = 1,
                      const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~AnalyticalSolution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double AnalyticalSolution<dim>::value(const Point<dim>    &/* p */,
                                      const unsigned int  /* component */) const
{
  double t = this->get_time();
  double result = 0.0;

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
double RightHandSide<dim>::value(const Point<dim>     & /* p */,
                                const unsigned int   /* component */) const
{
  double result = 1.0;
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
double NeumannBoundary<dim>::value(const Point<dim>   &/* p */,
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
  // constant velocity field (u,v) = (1,1)
  double value = 1.0;

  // circular velocity field (u,v) = (-y,x)
//  double value = 0.0;
//  if(component == 0)
//    value = - point[1];
//  else if(component == 1)
//    value = point[0];
//  else
//    AssertThrow(component <= 1, ExcMessage("Velocity field for 3-dimensional problem is not implemented!"));

  return value;
}

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>               &triangulation,
    unsigned int const                                      n_refine_space,
    std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor)
{
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);

  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
}

template<int dim>
void set_field_functions(std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  std_cxx11::shared_ptr<Function<dim> > velocity;
  velocity.reset(new VelocityField<dim>());

  field_functions->analytical_solution = analytical_solution;
  field_functions->right_hand_side = right_hand_side;
  field_functions->velocity = velocity;
}

template<int dim>
void set_analytical_solution(std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new AnalyticalSolution<dim>(1));
}


#endif /* APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_CONSTWINDCONSTRHS_H_ */
