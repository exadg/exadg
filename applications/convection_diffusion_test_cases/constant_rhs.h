/*
 * ConstantRHS.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONSTANT_RHS_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONSTANT_RHS_H_

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

void ConvDiff::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::ConvectionDiffusion;
  right_hand_side = true;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  diffusivity = 1.0;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  order_time_integrator = 4;
  start_with_low_order = true;
  calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  time_step_size = 1.0e-2;
  cfl = 0.2;
  diffusion_number = 0.01;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // polynomial degree
  degree = FE_DEGREE;
  degree_mapping = 1;

  // convective term
  numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::CG;
  solver_data = SolverData(1e4, 1.e-20, 1.e-6, 100);
  preconditioner = Preconditioner::Multigrid;
  mg_operator_type = MultigridOperatorType::ReactionDiffusion;
  // use default parameters of multigrid preconditioner

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  output_data.write_output = true;
  output_data.output_folder = "output_conv_diff/constant_rhs/";
  output_data.output_name = "constant_rhs";
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

  // set boundary indicator to 1 on all boundaries (Neumann BCs)
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
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

  double value (const Point<dim>   &/*p*/,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();

    double result = t;

    return result;
  }
};

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

  double value (const Point<dim>    &/*p*/,
                const unsigned int  /*component*/) const
  {
    double result = 1.0;
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
                const unsigned int  /*component*/) const
  {
    double value = 0.0;
    return value;
  }
};

template<int dim>
void set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  field_functions->analytical_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
  field_functions->velocity.reset(new VelocityField<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>(1));
}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONSTANT_RHS_H_ */
