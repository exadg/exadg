/*
 * BoundaryLayerProblem.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// prescribe value of phi at left and right boundary
// neumann boundaries at upper and lower boundary
// use constant advection velocity from left to right -> boundary layer

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 3;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 5;
const unsigned int REFINE_STEPS_SPACE_MAX = 5;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

// problem specific parameters
const double START_TIME = 0.0;
const double DIFFUSIVITY = 0.1;

void ConvDiff::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Steady;
  equation_type = EquationType::ConvectionDiffusion;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = 1.0; //8.0;
  diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDF;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  order_time_integrator = 2;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  time_step_size = 1.0e-1;
  cfl = 0.2;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  use_cell_based_face_loops = true;
  solver = Solver::GMRES;
  solver_data = SolverData(1e4, 1.e-20, 1.e-8, 100);
  preconditioner = Preconditioner::Multigrid;//Preconditioner::PointJacobi;
  mg_operator_type = MultigridOperatorType::ReactionConvectionDiffusion;
  multigrid_data.type = MultigridType::hMG;
  // MG smoother
  multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi; //Chebyshev;
  // MG smoother data
  multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data.smoother_data.iterations = 5;

  // MG coarse grid solver
  multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES; //AMG;
  
  update_preconditioner = false;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  output_data.write_output = true;
  output_data.output_folder = "output_conv_diff/boundary_layer_problem/";
  output_data.output_name = "boundary_layer_problem";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.degree = FE_DEGREE;

  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/20;
}


/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::Triangulation<dim>>       triangulation,
    unsigned int const                                  n_refine_space)
{
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      if ((std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12)||
         (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)
         || ((dim==3) && ((std::fabs(cell->face(face_number)->center()(2) - left ) < 1e-12) ||
                          (std::fabs(cell->face(face_number)->center()(2) - right) < 1e-12))))
        cell->face(face_number)->set_boundary_id (1);
    }
  }
  triangulation->refine_global(n_refine_space);
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

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    double result = 0.0;

    double phi_l = 1.0 , phi_r = 0.0;
    double U = 1.0, L = 2.0;
    double Pe = U*L/DIFFUSIVITY;
    if(t<(START_TIME + 1.0e-8))
      result = phi_l + (phi_r-phi_l)*(0.5+p[0]/L);
    else
      result = phi_l + (phi_r-phi_l)*(std::exp(Pe*p[0]/L)-std::exp(-Pe/2.0))/(std::exp(Pe/2.0)-std::exp(-Pe/2.0));

    return result;
  }
};

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

  double value (const Point<dim>    &/*p*/,
                const unsigned int  /*component*/) const
  {
    double result = 0.0;

  //  double right = 1.0;
  //  if( fabs(p[0]-right)<1.0e-12 )
  //    result = 1.0;

    return result;
  }
};

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

  double value (const Point<dim>    &/*p*/,
                const unsigned int  component = 0) const
  {
    double value = 0.0;

    if(component == 0)
      value = 1.0;

    return value;
  }
};

template<int dim>
void set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0,new Solution<dim>()));
  boundary_descriptor->neumann_bc.insert(pair(1,new NeumannBoundary<dim>()));
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  field_functions->analytical_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new VelocityField<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>(1));
}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_ */
