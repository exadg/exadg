/*
 * template.h
 *
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 4;
unsigned int const DEGREE_MAX = 4;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

namespace IncNS
{
void
set_input_parameters(InputParameters &param)
{
  (void)param;

  // Here, set all parameters differing from their default values as initialized in
  // IncNS::InputParameters::InputParameters()
}
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
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
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


/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number>>
construct_postprocessor(InputParameters const & param)
{
  (void)param;

  // these lines show exemplarily how the postprocessor is constructued
  PostProcessorData<dim> pp_data;

  std::shared_ptr<PostProcessorBase<dim, Number>> pp;
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
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(1));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param,
                        unsigned int const              scalar_index)
{
  (void)param;
  (void)scalar_index;

  PostProcessorData<dim> pp_data;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

} // namespace ConvDiff


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_ */
