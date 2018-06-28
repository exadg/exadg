/*
 * DiffusiveProblemHomogeneousDBC.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_

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
const unsigned int FE_DEGREE = 5;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 4;
const unsigned int REFINE_STEPS_SPACE_MAX = 4;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

// problem specific parameters
const double DIFFUSIVITY = 1.0e-1;

enum class BoundaryConditionType{
  HomogeneousDBC,
  HomogeneousNBC,
  HomogeneousNBCWithRHS
};

const BoundaryConditionType BOUNDARY_TYPE = BoundaryConditionType::HomogeneousNBCWithRHS;

const bool RIGHT_HAND_SIDE = (BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS) ? true : false;

void ConvDiff::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::Diffusion;
  right_hand_side = RIGHT_HAND_SIDE;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDF; //ExplRK; //BDF;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  order_time_integrator = 2;
  start_with_low_order = false;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepUserSpecified; //ConstTimeStepDiffusion; //ConstTimeStepUserSpecified;
  time_step_size = 1.0e-2;
  cfl_number = 0.1;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION
  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::PCG;
  abs_tol = 1.e-20;
  rel_tol = 1.e-6;
  max_iter = 1e4;
  preconditioner = Preconditioner::Multigrid; //BlockJacobi;
  mg_operator_type = MultigridOperatorType::ReactionDiffusion;
  multigrid_data.smoother = MultigridSmoother::Chebyshev; //GMRES;

  update_preconditioner = false;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  output_data.write_output = true;
  output_data.output_folder = "output_conv_diff/diffusive_problem_homogeneous_DBC/vtu/";
  output_data.output_name = "output";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.number_of_patches = FE_DEGREE;

  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  output_solver_info_every_timesteps = 1e1;
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
  double t = this->get_time();
  double result = 1.0;

  if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
  {
    for(int d=0;d<dim;d++)
      result *= std::cos(p[d]*numbers::PI/2.0);
    result *= std::exp(-0.5*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
  }
  else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
  {
    for(int d=0;d<dim;d++)
      result *= std::cos(p[d]*numbers::PI);
    result *= std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
  }
  else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
  {
    for(int d=0;d<dim;d++)
      result *= std::cos(p[d]*numbers::PI)+1.0;
    result *= std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

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
  double t = this->get_time();
  double result = 0.0;

  if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC ||
     BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
  {
    // do nothing, rhs=0
  }
  else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
  {
    for(int d=0;d<dim;++d)
      result += std::cos(p[d]*numbers::PI)+1;
    result *=  -std::pow(numbers::PI,2.0) * DIFFUSIVITY
               * std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

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

  if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
  {
    std::shared_ptr<Function<dim> > analytical_solution;
    analytical_solution.reset(new AnalyticalSolution<dim>());
    //problem with pure Dirichlet boundary conditions
    boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,analytical_solution));
  }
  else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC ||
          BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
  {
    std::shared_ptr<Function<dim> > neumann_bc;
    neumann_bc.reset(new NeumannBoundary<dim>());
    boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,neumann_bc));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());

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
  analytical_solution->solution.reset(new AnalyticalSolution<dim>(1));
}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_ */
