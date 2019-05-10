/*
 * diffusive_problem.h
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

namespace ConvDiff
{

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

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
  calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  time_step_size = 1.0e-2;
  cfl = 0.1;
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
  preconditioner = Preconditioner::Multigrid; //BlockJacobi;
  mg_operator_type = MultigridOperatorType::ReactionDiffusion;
  multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev; //GMRES;

  update_preconditioner = false;

  // NUMERICAL PARAMETERS
  runtime_optimization = false;

  // OUTPUT AND POSTPROCESSING
  output_data.write_output = true;
  output_data.output_folder = "output_conv_diff/diffusive_problem_homogeneous_DBC/vtu/";
  output_data.output_name = "output";
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

  triangulation->refine_global(n_refine_space);
}

/**************************************************************************************/
/*                                                                                    */
/*          FUNCTIONS (ANALYTICAL/INITIAL SOLUTION, BOUNDARY CONDITIONS, etc.)        */
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

  double value (const Point<dim>    &p,
                const unsigned int  /*component*/) const
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
};

template<int dim>
void set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
  {
    boundary_descriptor->dirichlet_bc.insert(pair(0,new Solution<dim>()));
  }
  else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC ||
          BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
  {
    boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  field_functions->analytical_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>(1));
}

}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_ */
