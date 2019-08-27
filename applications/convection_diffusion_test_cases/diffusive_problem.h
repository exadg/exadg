/*
 * diffusive_problem.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 5;
unsigned int const DEGREE_MAX = 5;

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
double const DIFFUSIVITY = 1.0e-1;

double const START_TIME = 0.0;
double const END_TIME = 1.0;

enum class BoundaryConditionType{
  HomogeneousDBC,
  HomogeneousNBC,
  HomogeneousNBCWithRHS
};

const BoundaryConditionType BOUNDARY_TYPE = BoundaryConditionType::HomogeneousDBC;

const bool RIGHT_HAND_SIDE = (BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS) ? true : false;

namespace ConvDiff
{
void
set_input_parameters(ConvDiff::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::Diffusion;
  param.right_hand_side = RIGHT_HAND_SIDE;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
  param.time_integrator_rk = TimeIntegratorRK::ExplRK3Stage7Reg2;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.order_time_integrator = 3;
  param.start_with_low_order = false;
  param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  param.time_step_size = 1.0e-3;
  param.cfl = 0.1;
  param.diffusion_number = 0.04; //0.01;
  param.dt_refinements = REFINE_TIME_MIN;

  // SPATIAL DISCRETIZATION

  // triangulation
  param.triangulation_type = TriangulationType::Distributed;

  // polynomial degree
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Affine;

  // h-refinements
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  param.IP_factor = 1.0;

  // SOLVER
  param.solver = Solver::CG; // use FGMRES for elementwise iterative block Jacobi type preconditioners
  param.solver_data = SolverData(1e4, 1.e-20, 1.e-6, 100);
  param.preconditioner = Preconditioner::InverseMassMatrix; //BlockJacobi; //Multigrid;
  param.mg_operator_type = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev; //GMRES;
  param.implement_block_diagonal_preconditioner_matrix_free = true;
  param.solver_block_diagonal = Elementwise::Solver::CG;
  param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal = SolverData(1000, 1.e-12, 1.e-2, 1000);
  param.use_cell_based_face_loops = true;
  param.update_preconditioner = false;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = END_TIME-START_TIME;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
}
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::TriangulationBase<dim>>       triangulation,
    unsigned int const                                  n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >              &periodic_faces)
{
  (void)periodic_faces;

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

namespace ConvDiff
{

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
  field_functions->initial_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = "output_conv_diff/diffusive_problem_homogeneous_DBC/vtu/";
  pp_data.output_data.output_name = "output";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.degree = param.degree;

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
  pp_data.error_data.error_calc_start_time = param.start_time;
  pp_data.error_data.error_calc_interval_time = (param.end_time-param.start_time)/20;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DIFFUSIVE_PROBLEM_H_ */
