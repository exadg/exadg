/*
 * rotating_hill.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 5;
unsigned int const DEGREE_MAX = 5;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
double const START_TIME = 0.0;
double const END_TIME = 1.0;

namespace ConvDiff
{
void
set_input_parameters(ConvDiff::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::Convection;
  param.analytical_velocity_field = true;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = 0.0;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::ExplRK; //BDF; //ExplRK;

  // Explicit RK
  param.time_integrator_rk = TimeIntegratorRK::ExplRK3Stage7Reg2;

  // BDF
  param.order_time_integrator = 2; // instabilities for BDF 3 and 4
  param.start_with_low_order = false;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit; //ExplicitOIF;
  param.time_integrator_oif = TimeIntegratorRK::ExplRK2Stage2; //ExplRK3Stage7Reg2; //ExplRK4Stage8Reg2;

  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = false;
  param.time_step_size = 1.e-2;
  param.cfl_oif = 0.5;
  param.cfl = param.cfl_oif * 1.0;
  param.diffusion_number = 0.01;
  param.exponent_fe_degree_convection = 2.0;
  param.exponent_fe_degree_diffusion = 3.0;
  param.c_eff = 1.0e0;
  param.dt_refinements = REFINE_TIME_MIN;

  // restart
  param.restart_data.write_restart = false;
  param.restart_data.filename = "output_conv_diff/rotating_hill";
  param.restart_data.interval_time = 0.4;


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
  param.solver_data = SolverData(1e3, 1.e-20, 1.e-8, 100);
  param.preconditioner = Preconditioner::Multigrid; //None; //InverseMassMatrix; //PointJacobi; //BlockJacobi; //Multigrid;
  param.update_preconditioner = true;

  // BlockJacobi (these parameters are also relevant if used as a smoother in multigrid)
  param.implement_block_diagonal_preconditioner_matrix_free = true;
  param.solver_block_diagonal = Elementwise::Solver::GMRES;
  param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal = SolverData(1000,1.e-12,1.e-2,1000);

  // Multigrid
  param.mg_operator_type = MultigridOperatorType::ReactionConvection;
  param.multigrid_data.type = MultigridType::hMG;

  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;
  param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data.smoother_data.iterations = 5;
  param.multigrid_data.smoother_data.relaxation_factor = 0.8;

  // MG coarse grid solver
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/20;

  // NUMERICAL PARAMETERS
  param.use_cell_based_face_loops = true;
  param.store_analytical_velocity_in_dof_vector = false;
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
  Solution (const unsigned int  n_components = 1,
            const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();

    double radius = 0.5;
    double omega = 2.0*numbers::PI;
    double center_x = -radius*std::sin(omega*t);
    double center_y = +radius*std::cos(omega*t);
    double result = std::exp(-50*pow(p[0]-center_x,2.0)-50*pow(p[1]-center_y,2.0));

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

  double value (const Point<dim>    &point,
                const unsigned int  component = 0) const
  {
    double value = 0.0;

    if(component == 0)
      value = -point[1]*2.0*numbers::PI;
    else if(component ==1)
      value =  point[0]*2.0*numbers::PI;

    return value;
  }
};

namespace ConvDiff
{

template<int dim>
void set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(pair(0,new Solution<dim>()));
}

template<int dim>
void set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution.reset(new Solution<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new VelocityField<dim>());
}

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = "output_conv_diff/";
  pp_data.output_data.output_name = "rotating_hill";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.degree = param.degree;

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
  pp_data.error_data.calculate_relative_errors = true;
  pp_data.error_data.error_calc_start_time = param.start_time;
  pp_data.error_data.error_calc_interval_time = (param.end_time-param.start_time)/20;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_ */
