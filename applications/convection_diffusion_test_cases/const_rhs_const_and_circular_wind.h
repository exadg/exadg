/*
 * ConstWindConstRHS.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// constant source term inside rectangular domain
// pure Dirichlet boundary conditions (homogeneous)
// use constant or circular advection velocity

// convergence studies in space or time
unsigned int const DEGREE_MIN = 4;
unsigned int const DEGREE_MAX = 4;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
const double DIFFUSIVITY = 1.0e0;

double const START_TIME = 0.0;
double const END_TIME = 1.0;

enum class VelocityType { Constant, Circular, CircularZeroAtBoundary };
VelocityType const VELOCITY_TYPE = VelocityType::Constant; //CircularZeroAtBoundary; //Constant; //Circular;

namespace ConvDiff
{
void
set_input_parameters(ConvDiff::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Steady;
  param.equation_type = EquationType::ConvectionDiffusion;
  param.analytical_velocity_field = true;
  param.right_hand_side = true;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
  param.time_integrator_rk = TimeIntegratorRK::ExplRK3Stage7Reg2;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  param.time_step_size = 1.0e-1;
  param.cfl = 0.2;
  param.diffusion_number = 0.01;
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
  param.solver = Solver::GMRES;
  param.solver_data = SolverData(1e4,1.e-20, 1.e-8, 100);
  param.preconditioner = Preconditioner::Multigrid; //PointJacobi; //BlockJacobi;
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;
  param.solver_block_diagonal = Elementwise::Solver::GMRES;
  param.update_preconditioner = true;

  param.multigrid_data.type = MultigridType::phMG;
  param.mg_operator_type = MultigridOperatorType::ReactionConvectionDiffusion;
  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;

  // MG smoother data: Chebyshev smoother
//  param.multigrid_data.smoother_data.iterations = 3;

  // MG smoother data: GMRES smoother, CG smoother
  param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data.smoother_data.iterations = 4;

  // MG smoother data: Jacobi smoother
//  param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
//  param.multigrid_data.smoother_data.iterations = 5;
//  param.multigrid_data.smoother_data.relaxation_factor = 0.8;

  // MG coarse grid solver
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::AMG; //GMRES;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/10;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
  param.store_analytical_velocity_in_dof_vector = true;
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
  Solution (const unsigned int n_components = 1,
            const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>   &/*p*/,
                const unsigned int /*component*/) const
  {
    double result = 0.0;

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
  RightHandSide (const unsigned int n_components = 1,
                 const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>  &/*p*/,
                const unsigned int /*component*/) const
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

  double value (const Point<dim>   &point,
                const unsigned int component = 0) const
  {
    double value = 0.0;

    if(VELOCITY_TYPE == VelocityType::Constant)
    {
      // constant velocity field (u,v) = (1,1)
      value = 1.0;
    }
    else if(VELOCITY_TYPE == VelocityType::Circular)
    {
      // circular velocity field (u,v) = (-y,x)
      if(component == 0)
        value = - point[1];
      else if(component == 1)
        value = point[0];
      else
        AssertThrow(component <= 1, ExcMessage("Velocity field for 3-dimensional problem is not implemented!"));
    }
    else if(VELOCITY_TYPE == VelocityType::CircularZeroAtBoundary)
    {
      const double pi = numbers::PI;
      double sinx = std::sin(pi*point[0]);
      double siny = std::sin(pi*point[1]);
      double sin2x = std::sin(2.*pi*point[0]);
      double sin2y = std::sin(2.*pi*point[1]);
      if (component == 0)
        value = pi*sin2y*std::pow(sinx,2.);
      else if (component == 1)
        value = -pi*sin2x*std::pow(siny,2.);
    }
    else
    {
      AssertThrow(false, ExcMessage("Invalid type of velocity field prescribed for this problem."));
    }

    return value;
  }
};

namespace ConvDiff
{

template<int dim>
void set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0,new Solution<dim>()));
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
  field_functions->velocity.reset(new VelocityField<dim>());
}

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = false;
  pp_data.output_data.output_folder = "output_conv_diff/";
  pp_data.output_data.output_name = "const_wind_const_rhs";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = param.end_time-param.start_time;
  pp_data.output_data.degree = param.degree;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_ */
