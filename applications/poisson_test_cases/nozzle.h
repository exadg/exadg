
#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../grid_tools/deformed_cube_manifold.h"

// nozzle geometry
#include "../grid_tools/fda_benchmark/nozzle.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space
unsigned int const DEGREE_MIN = 1;
unsigned int const DEGREE_MAX = 15;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = 0;

// problem specific parameters
std::string OUTPUT_FOLDER     = "output/poisson/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME       = "nozzle_refine0";

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
  param.mapping = MappingType::Cubic; //Isoparametric;
  param.spatial_discretization = SpatialDiscretization::DG;
  param.IP_factor = 1.0e0;

  // SOLVER
  param.solver = Poisson::Solver::CG;
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
                                   Triangulation<dim>::cell_iterator> >         &periodic_faces)
{
  create_grid_and_set_boundary_ids_nozzle(triangulation, n_refine_space, periodic_faces);
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
  value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
  {
    double result = 1.0;

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

  // inflow
  boundary_descriptor->dirichlet_bc.insert(pair(1, new Solution<dim>()));

  // outflow
  boundary_descriptor->dirichlet_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));

  // walls
  boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
}

template<int dim, typename Number>
std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number> >
construct_postprocessor(Poisson::InputParameters const &param)
{
  ConvDiff::PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.degree = param.degree;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim,Number> > pp;
  pp.reset(new ConvDiff::PostProcessor<dim,Number>(pp_data));

  return pp;
}

}
