/*
 * DeformingHill.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DEFORMINGHILL_H_
#define APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DEFORMINGHILL_H_

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 2;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 4;
const unsigned int REFINE_STEPS_SPACE_MAX = 4;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

// problem specific parameters
const double END_TIME = 1.0; //increase end_time for larger deformations of the hill

void InputParametersConvDiff::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::Convection;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME;
  diffusivity = 0.0;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  order_time_integrator = 4;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  time_step_size = 1.0e-4;
  cfl_number = 0.2;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION
  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // SOLVER
  solver = Solver::GMRES;
  abs_tol = 1.e-20;
  rel_tol = 1.e-6;
  max_iter = 1e4;
  preconditioner = Preconditioner::InverseMassMatrix;

  // viscous term
  IP_factor = 1.0;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  write_output = true;
  output_prefix = "deforming_hill";
  output_start_time = start_time;
  output_interval_time = (end_time-start_time)/20;

  //analytical solution only available at t = start_time and t = end_time
  analytical_solution_available = true;
  error_calc_start_time = start_time;
  error_calc_interval_time = end_time-start_time;

  output_solver_info_every_timesteps = 1e6;
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
double AnalyticalSolution<dim>::value(const Point<dim>    &p,
                                      const unsigned int  /* component */) const
{
  // The analytical solution is only known at t = start_time and t = end_time

  double center_x = 0.5;
  double center_y = 0.75;
  double factor = 50.0;
  double result = std::exp(-factor*(pow(p[0]-center_x,2.0)+pow(p[1]-center_y,2.0)));

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
double RightHandSide<dim>::value(const Point<dim>     &p,
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
  double t = this->get_time();

  if(component == 0)
    value =  4.0 * std::sin(numbers::PI*point[0])*std::sin(numbers::PI*point[0])
                 * std::sin(numbers::PI*point[1])*std::cos(numbers::PI*point[1])
                 * std::cos(numbers::PI*t/END_TIME);
  else if(component ==1)
    value = -4.0 * std::sin(numbers::PI*point[0])*std::cos(numbers::PI*point[0])
                 * std::sin(numbers::PI*point[1])*std::sin(numbers::PI*point[1])
                 * std::cos(numbers::PI*t/END_TIME);

  return value;
}

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_conditions(parallel::distributed::Triangulation<dim>               &triangulation,
                                             unsigned int const                                      n_refine_space,
                                             std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor)
{
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());

  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(
      std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
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


#endif /* APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DEFORMINGHILL_H_ */
