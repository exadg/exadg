
#include "../../include/convection_diffusion/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space
unsigned int const DEGREE_MIN = 1;
unsigned int const DEGREE_MAX = 15;

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

// problem specific parameters
std::string OUTPUT_FOLDER     = "output/poisson/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME       = "slit";

namespace Poisson
{
void
set_input_parameters(Poisson::InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.right_hand_side = false;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Isoparametric;
  param.spatial_discretization = SpatialDiscretization::DG;
  param.IP_factor = 1.0e0;

  // SOLVER
  param.solver = Poisson::Solver::CG;
  param.solver_data.abs_tol = 1.e-20;
  param.solver_data.rel_tol = 1.e-10;
  param.solver_data.max_iter = 1e4;
  param.compute_performance_metrics = true;
  param.preconditioner = Preconditioner::Multigrid;
  param.multigrid_data.type = MultigridType::hcpMG;
  param.multigrid_data.p_sequence = PSequenceType::Bisect;
  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data.smoother_data.iterations = 5;
  // MG coarse grid solver
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
  param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-6;
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
                                   Triangulation<dim>::cell_iterator> >         &/*periodic_faces*/)
{
  const double length = 1.0;
  const double left = -length, right = length;

  GridGenerator::hyper_cube_slit (*triangulation, left, right);

  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

#include <deal.II/base/function_lib.h>

template <int dim>
class Solution : public Functions::SlitSingularityFunction<dim>
{
public:
  Solution() : Functions::SlitSingularityFunction<dim>() {}
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
  ConvDiff::PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = false;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.write_higher_order = true;
  pp_data.output_data.degree = param.degree;

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.analytical_solution.reset(new Solution<dim>());

  std::shared_ptr<ConvDiff::PostProcessorBase<dim,Number> > pp;
  pp.reset(new ConvDiff::PostProcessor<dim,Number>(pp_data));

  return pp;
}

}
