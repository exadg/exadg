/*
 * DiffusiveProblemHomogeneousNBC2.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DIFFUSIVEPROBLEMHOMOGENEOUSNBC2_H_
#define APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DIFFUSIVEPROBLEMHOMOGENEOUSNBC2_H_


/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// diffusive problem with pure Neumann boundary conditions (homogeneous)

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 2;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = 3;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

// problem specific parameters
const double DIFFUSIVITY = 0.2;

void InputParametersConvDiff::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationTypeConvDiff::Diffusion;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  order_time_integrator = 4;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFLAndDiffusion;
  time_step_size = 1.0e-2;
  cfl_number = 0.2;
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
  preconditioner = Preconditioner::GeometricMultigrid;
  // use default parameters of multigrid preconditioner

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  write_output = true;
  output_prefix = "diffusive_problem_homogeneous_NBC_2";
  output_start_time = start_time;
  output_interval_time = (end_time-start_time)/20;

  analytical_solution_available = true;
  error_calc_start_time = start_time;
  error_calc_interval_time = output_interval_time;

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
  double t = this->get_time();
  double result = 1.0;

  for(int d=0;d<dim;d++)
    result *= std::cos(p[d]*numbers::PI);
  result *= std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);

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

  // set boundary indicator to 1 on all boundaries (Neumann BCs)
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // apply Neumann BC on all boundaries
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12) ||
          (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
       cell->face(face_number)->set_boundary_id(1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>());
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
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



#endif /* APPLICATIONS_CONVECTIONDIFFUSIONTESTCASES_DIFFUSIVEPROBLEMHOMOGENEOUSNBC2_H_ */
