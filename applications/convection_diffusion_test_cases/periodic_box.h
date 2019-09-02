/*
 * template.h
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../grid_tools/periodic_box.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 1;
unsigned int const DEGREE_MAX = 10;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = 0;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

namespace ConvDiff
{
void
set_input_parameters(ConvDiff::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::ConvectionDiffusion;
  param.right_hand_side = false;
  // Note: set parameter store_analytical_velocity_in_dof_vector to test different implementation variants
  param.analytical_velocity_field = true;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 1.0;
  param.diffusivity = 1.0;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  param.time_step_size = 1.e-2;

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
  param.preconditioner = Preconditioner::None;

  // NUMERICAL PARAMETERS
  param.use_cell_based_face_loops = false;
  param.use_combined_operator = true;
  param.store_analytical_velocity_in_dof_vector = true; // true; // false;
}
}


/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                                n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >            &periodic_faces,
                                 unsigned int const                                n_subdivisions = 1)
{
  double const left = -1.0, right = 1.0;
  double const deformation = 0.1;

  bool curvilinear_mesh = false;
  if(MESH_TYPE == MeshType::Cartesian)
  {
    // do nothing
  }
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    curvilinear_mesh = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  create_periodic_box(triangulation,
                      n_refine_space,
                      periodic_faces,
                      n_subdivisions,
                      left,
                      right,
                      curvilinear_mesh,
                      deformation);
}


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

//  Example for a user defined function
template<int dim>
class Velocity : public Function<dim>
{
public:
  Velocity(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    return p[component];
  }
};

namespace ConvDiff
{

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  // these lines show exemplarily how the boundary descriptors are filled
  boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
}


template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions)
{
  // these lines show exemplarily how the field functions are filled
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Velocity<dim>(dim));
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<ConvDiff::AnalyticalSolution<dim>> analytical_solution)
{
  // these lines show exemplarily how the analytical solution is filled
  analytical_solution->solution.reset(new Functions::ZeroFunction<dim>(1));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param)
{
  (void)param;

  PostProcessorData<dim> pp_data;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

} // namespace ConvDiff

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_ */
