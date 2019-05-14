/*
 * BoundaryLayerProblem.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// prescribe value of phi at left and right boundary
// neumann boundaries at upper and lower boundary
// use constant advection velocity from left to right -> boundary layer

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 5;
unsigned int const REFINE_SPACE_MAX = 5;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
double const DIFFUSIVITY = 1.0e-1;

double const START_TIME = 0.0;
double const END_TIME = 1.0; // 8.0;

namespace ConvDiff
{
void
set_input_parameters(ConvDiff::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Steady;
  param.equation_type = EquationType::ConvectionDiffusion;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
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
  param.use_cell_based_face_loops = true;
  param.solver = Solver::GMRES;
  param.solver_data = SolverData(1e4, 1.e-20, 1.e-8, 100);
  param.preconditioner = Preconditioner::Multigrid;//Preconditioner::PointJacobi;
  param.mg_operator_type = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data.type = MultigridType::hMG;
  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi; //Chebyshev;
  // MG smoother data
  param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data.smoother_data.iterations = 5;

  // MG coarse grid solver
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES; //AMG;
  
  param.update_preconditioner = false;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/20;

  // NUMERICAL PARAMETERS
  param.runtime_optimization = false;
}
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::Triangulation<dim>>       triangulation,
    unsigned int const                                  n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >              &periodic_faces)
{
  (void)periodic_faces;

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

namespace ConvDiff
{

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
  field_functions->initial_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new VelocityField<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>(1));
}

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor()
{
  PostProcessorData pp_data;
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = "output_conv_diff/boundary_layer_problem/";
  pp_data.output_data.output_name = "boundary_layer_problem";
  pp_data.output_data.output_start_time = START_TIME;
  pp_data.output_data.output_interval_time = (END_TIME-START_TIME)/20;
  pp_data.output_data.degree = DEGREE_MIN;

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.error_calc_start_time = START_TIME;
  pp_data.error_data.error_calc_interval_time = (END_TIME-START_TIME)/20;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_PROBLEM_H_ */
