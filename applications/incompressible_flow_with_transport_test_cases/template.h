/*
 * template.h
 *
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// TODO remove all global variables

// single or double precision?
// typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 4;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY - 1;
unsigned int const FE_DEGREE_SCALAR = FE_DEGREE_VELOCITY;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

template<int dim>
void
IncNS::InputParameters<dim>::set_input_parameters()
{
  // Here, set all parameters differing from their default values as initialized in
  // IncNS::InputParameters<dim>::InputParameters()
}

namespace ConvDiff
{
void 
set_input_parameters(InputParameters &param, unsigned int const scalar_index)
{
  (void)param;
  (void)scalar_index;

  // Here, set all parameters differing from their default values as initialized in
  // ConvDiff::InputParameters::InputParameters()
}
}


/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(
  std::shared_ptr<parallel::Triangulation<dim>> triangulation,
  unsigned int const                            n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_faces)
{
  // to avoid warnings (unused variable) use ...
  (void)triangulation;
  (void)n_refine_space;
  (void)periodic_faces;
}


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

//  Example for a user defined function
template<int dim>
class MyFunction : public Function<dim>
{
public:
  MyFunction(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)p;
    (void)component;

    return 0.0;
  }
};

namespace IncNS
{
template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                        std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  // these lines show exemplarily how the boundary descriptors are filled

  // velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

  // pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  // these lines show exemplarily how the field functions are filled
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim>> analytical_solution)
{
  // these lines show exemplarily how the analytical solution is filled
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}


/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number>>
construct_postprocessor(InputParameters<dim> const & param)
{
  // these lines show exemplarily how the postprocessor is constructued
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data  = param.error_data;

  std::shared_ptr<PostProcessor<dim, Number>> pp;
  pp.reset(new PostProcessor<dim, Number>(pp_data));

  return pp;
}

} // namespace IncNS


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

namespace ConvDiff
{

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor, 
                        unsigned int const                                 scalar_index = 0)
{
  (void)scalar_index;

  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  // these lines show exemplarily how the boundary descriptors are filled
  boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
}


template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions, 
                    unsigned int const                             scalar_index = 0)
{
  (void)scalar_index;

  // these lines show exemplarily how the field functions are filled
  field_functions->analytical_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(1));
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim>> analytical_solution, 
                        unsigned int const                                 scalar_index = 0)
{
  (void)scalar_index;

  // these lines show exemplarily how the analytical solution is filled
  analytical_solution->solution.reset(new Functions::ZeroFunction<dim>(1));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessor<dim, Number> >
construct_postprocessor()
{
  PostProcessorData pp_data;
  pp_data.output_data = OutputData();
  pp_data.error_data  = ErrorCalculationData();

  std::shared_ptr<PostProcessor<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

} // namespace ConvDiff


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_ */
