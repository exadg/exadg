
#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../grid_tools/deformed_cube_manifold.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space
unsigned int const DEGREE_MIN = 1;
unsigned int const DEGREE_MAX = 15;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

// problem specific parameters
std::string OUTPUT_FOLDER     = "output/poisson/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME       = "sine";

double const FREQUENCY = 3.0*numbers::PI;

enum class MeshType{
  Cartesian,
  Curvilinear
};

MeshType const MESH_TYPE = MeshType::Curvilinear;

namespace Poisson
{
void
set_input_parameters(Poisson::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.right_hand_side = true;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Cubic; //Isoparametric;
  param.spatial_discretization = SpatialDiscretization::DG;
  param.IP_factor = 1.0e0;

  // SOLVER
  param.solver = Solver::CG;
  param.solver_data.abs_tol = 1.e-20;
  param.solver_data.rel_tol = 1.e-10;
  param.solver_data.max_iter = 1e4;
  param.compute_performance_metrics = true;
  param.preconditioner = Preconditioner::Multigrid;
  param.multigrid_data.type = MultigridType::cphMG;
  param.multigrid_data.p_sequence = PSequenceType::Bisect;
  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data.smoother_data.iterations = 5;
  param.multigrid_data.smoother_data.smoothing_range = 20;
  // MG coarse grid solver
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
  param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
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
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >         &periodic_faces,
                                 unsigned int const n_subdivisions = 2)
{
  (void)periodic_faces;

  const double length = 1.0;
  const double left = -length, right = length;
  GridGenerator::subdivided_hyper_cube(*triangulation,n_subdivisions,left,right);

  if(MESH_TYPE == MeshType::Cartesian)
  {
    // do nothing
  }
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    double const deformation = 0.15;
    unsigned int const frequency = 2;
    DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
    triangulation->set_all_manifold_ids(1);
    triangulation->set_manifold(1, manifold);

    std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

    for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
        cell != triangulation->end(); ++cell)
    {
      for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        if (vertex_touched[cell->vertex_index(v)]==false)
        {
          Point<dim> &vertex = cell->vertex(v);
          Point<dim> new_point = manifold.push_forward(vertex);
          vertex = new_point;
          vertex_touched[cell->vertex_index(v)] = true;
        }
      }
    }
  }

  triangulation->refine_global(n_refine_space);
}


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double result = 1.0;
    for(unsigned int d=0; d<dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

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
  RightHandSide(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /* component */) const
  {
    double result = FREQUENCY * FREQUENCY * dim;
    for(unsigned int d=0; d<dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};

namespace Poisson
{

template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim, typename Number>
std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number> >
construct_postprocessor(Poisson::InputParameters const &param)
{
  ConvDiff::PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = false;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.write_higher_order = true;
  pp_data.output_data.degree = param.degree;

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.analytical_solution.reset(new Solution<dim>());
  pp_data.error_data.calculate_relative_errors = true;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim,Number> > pp;
  pp.reset(new ConvDiff::PostProcessor<dim,Number>(pp_data));

  return pp;
}

}
