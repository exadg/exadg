
#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../grid_tools/deformed_cube_manifold.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space
unsigned int const DEGREE_MIN = 7;
unsigned int const DEGREE_MAX = 7;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

// problem specific parameters
std::string OUTPUT_FOLDER     = "output/poisson/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME       = "cosinus";

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
  param.mapping = MappingType::Isoparametric;
  param.spatial_discretization = SpatialDiscretization::DG;
  param.IP_factor = 1.0;

  // SOLVER
  param.solver = Solver::CG;
  param.solver_data = SolverData(1e4, 1.e-20, 1.e-8);
  param.preconditioner = Preconditioner::Multigrid;
  param.multigrid_data.type = MultigridType::pMG;
  param.multigrid_data.dg_to_cg_transfer = DG_To_CG_Transfer::Fine;
  // MG smoother
  param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
  // MG smoother data
  param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
}

}


/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::Triangulation<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >         &periodic_faces)
{
  // hypercube: [left,right]^dim
  const double left = -0.5 * numbers::PI, right = +0.5 * numbers::PI;
  const double deformation = +0.1, frequnency = +2.0;
  GridGenerator::hyper_cube(*triangulation, left, right);

  static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
  triangulation->set_all_manifold_ids(1);
  triangulation->set_manifold(1, manifold);
  triangulation->refine_global(n_refine_space);

  for (auto cell : (*triangulation))
  {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (cell.face(face)->at_boundary())
      {
        if(std::abs(cell.face(face)->center()(1) - left) < 1e-12)
            cell.face(face)->set_all_boundary_ids(2);
        else if(std::abs(cell.face(face)->center()(1) - right) < 1e-12)
            cell.face(face)->set_all_boundary_ids(3);
      }
    }
  }

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
  triangulation->add_periodicity(periodic_faces);
}


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double result = 0.1 * p[0];
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
    const double coef = 1.0;
    double       temp = 1;
    for(int i = 0; i < dim; i++)
      temp *= std::cos(p[i]);
    return temp * dim * coef;
  }
};

namespace Poisson
{

template<int dim>
void
set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0, new DirichletBC<dim>()));

//  boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
{
  field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim>> analytical_solution)
{
  analytical_solution->solution.reset(new Functions::ZeroFunction<dim>(1));
}

template<int dim, typename Number>
std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number> >
construct_postprocessor()
{
  ConvDiff::PostProcessorData pp_data;
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;

  pp_data.error_data = ErrorCalculationData();

  std::shared_ptr<ConvDiff::PostProcessorBase<dim,Number> > pp;
  pp.reset(new ConvDiff::PostProcessor<dim,Number>(pp_data));

  return pp;
}

}
