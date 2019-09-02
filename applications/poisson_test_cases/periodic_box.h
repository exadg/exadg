
#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../grid_tools/periodic_box.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = 0;

// problem specific parameters

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

namespace Poisson
{
void
set_input_parameters(Poisson::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.right_hand_side = false;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Affine; //Cubic;
  param.spatial_discretization = SpatialDiscretization::DG;
  param.IP_factor = 1.0e0;

  // SOLVER
  param.solver = Poisson::Solver::CG;
  param.preconditioner = Preconditioner::None;
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

namespace Poisson
{

template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
{
  (void)boundary_descriptor;
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number> >
construct_postprocessor(Poisson::InputParameters const &param)
{
  (void)param;

  ConvDiff::PostProcessorData<dim> pp_data;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim,Number> > pp;
  pp.reset(new ConvDiff::PostProcessor<dim,Number>(pp_data));

  return pp;
}

}
