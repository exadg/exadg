/*
 * propagation_sine_wave.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// test case for purely convective problem
// sine wave that is advected from left to right by a constant velocity field

// convergence studies in space or time
unsigned int const DEGREE_MIN = 7;
unsigned int const DEGREE_MAX = 7;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
double const START_TIME = 0.0;
double const END_TIME = 8.0;

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
  param.time_integrator_rk = TimeIntegratorRK::ExplRK3Stage7Reg2;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //ExplicitOIF; //Explicit;
  param.time_integrator_oif = TimeIntegratorRK::ExplRK3Stage7Reg2;
  param.order_time_integrator = 2;
  param.start_with_low_order = false;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.time_step_size = 1.0e-1;
  param.cfl = 0.2;
  param.cfl_oif = param.cfl/1.0;
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
  param.solver_data = SolverData(1e4, 1.e-20, 1.e-6, 100);
  param.preconditioner = Preconditioner::InverseMassMatrix;
  // use default parameters of multigrid preconditioner

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/20;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
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

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      //use outflow boundary at right boundary
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12))
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

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();

    double result = std::sin(numbers::PI*(p[0]-t));

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
  boundary_descriptor->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
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
  pp_data.output_data.write_output = false;
  pp_data.output_data.output_folder = "output_conv_diff/propagating_sine_wave/";
  pp_data.output_data.output_name = "propagating_sine_wave";
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

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_ */
