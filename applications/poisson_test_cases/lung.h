#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include "../incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "lung/lung_environment.h"
#include "lung/lung_grid.h"

/******************************************************************************/
/*                                                                            */
/*                             INPUT PARAMETERS                               */
/*                                                                            */
/******************************************************************************/

const unsigned int DIMENSION              = 3;
const unsigned int FE_DEGREE              = 4;
const unsigned int FE_DEGREE_MIN          = 1;
const unsigned int FE_DEGREE_MAX          = 7;
const unsigned int REFINE_STEPS           = 2;
const unsigned int REFINE_STEPS_SPACE_MIN = 5;
const unsigned int REFINE_STEPS_SPACE_MAX = 5;

std::string OUTPUT_FOLDER     = "output/poisson_gaussian/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME       = "cosinus";

void
Poisson::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  right_hand_side = true;

  // PHYSICAL QUANTITIES

  // TEMPORAL DISCRETIZATION

  // SPATIAL DISCRETIZATION
  IP_factor = 1.0;

  // SOLVER
  solver               = Poisson::Solver::CG;
  solver_data.abs_tol  = 1.e-20;
  solver_data.rel_tol  = 1.e-8;
  solver_data.max_iter = 1e4;

  preconditioner = Preconditioner::Multigrid;
  // MG smoother
  // multigrid_data.smoother = MultigridSmoother::Chebyshev;
  // MG smoother data
  // multigrid_data.chebyshev_smoother_data.smoother_smoothing_range = 30;
  // multigrid_data.chebyshev_smoother_data.smoother_poly_degree     = 5;

  // multigrid_data.gmres_smoother_data.preconditioner       = PreconditionerGMRESSmoother::None;
  // multigrid_data.gmres_smoother_data.number_of_iterations = 5;
  // MG coarse grid solver
  // multigrid_data.coarse_solver = MultigridCoarseGridSolver::GMRES_PointJacobi; //
  // GMRES_PointJacobi;
  multigrid_data.type = MultigridType::pMG;

  multigrid_data.dg_to_cg_transfer = DG_To_CG_Transfer::Coarse;

  // write output for visualization of results
  output_data.write_output  = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name   = OUTPUT_NAME;
}

/******************************************************************************/
/*                                                                            */
/* FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.) */
/*                                                                            */
/*                                                                            */
/******************************************************************************/

template<int dim>
class CoefficientFunction : public Function<dim>
{
public:
  CoefficientFunction() : Function<dim>(1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int c = 0) const
  {
    (void)c;
    return value<double>(p);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int c = 0) const
  {
    (void)c;
    (void)p;
    Tensor<1, dim> grad;
#if MESH_TYPE == 1
    for(unsigned int d = 0; d < dim; ++d)
      grad[d] = -4e6 * numbers::PI * std::cos(2. * numbers::PI * p[d] + 0.1 * (d + 1)) *
                std::sin(2. * numbers::PI * p[d] + 0.1 * (d + 1));
#endif
    return grad;
  }

  template<typename Number>
  Number
  value(const dealii::Point<dim, Number> & p) const
  {
    (void)p;
    Number value;
    value = 1;
#if MESH_TYPE == 1
    for(unsigned int d = 0; d < dim; ++d)
    {
      const Number cosp = std::cos(2. * dealii::numbers::PI * p[d] + 0.1 * (d + 1));
      value += cosp * cosp * 1e6;
    }
#endif
    return value;
  }
};

/*
 *  Analytical solution
 */

#if MESH_TYPE == 2

template<int dim>
class Solution : public Functions::SlitSingularityFunction<dim>
{
public:
  Solution() : Functions::SlitSingularityFunction<dim>()
  {
  }
};

#else

template<int dim>
class SolutionBase
{
protected:
  static const unsigned int n_source_centers = 3;
  static const Point<dim>   source_centers[n_source_centers];
  static const double       width;
};


template<>
const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
  {Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)};


template<>
const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  {Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)};

template<>
const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] =
  {Point<3>(-0.5, +0.5, 0.25), Point<3>(-0.6, -0.5, -0.125), Point<3>(+0.5, -0.5, 0.5)};

template<int dim>
const double SolutionBase<dim>::width = 1. / 5.;

template<int dim>
class AnalyticalSolution : public Function<dim>, protected SolutionBase<dim>
{
public:
  AnalyticalSolution() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const;
};

template<int dim>
double
AnalyticalSolution<dim>::value(const Point<dim> & p, const unsigned int) const
{
  double return_value = 0;
  for(unsigned int i = 0; i < this->n_source_centers; ++i)
  {
    const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
    return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
  }

  return 1.0;

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2. * numbers::PI) * this->width);
}

template<int dim>
Tensor<1, dim>
AnalyticalSolution<dim>::gradient(const Point<dim> & p, const unsigned int) const
{
  Tensor<1, dim> return_value;

  for(unsigned int i = 0; i < this->n_source_centers; ++i)
  {
    const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

    return_value +=
      (-2 / (this->width * this->width) *
       std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * numbers::PI) * this->width);
}


#endif

/*
 *  Right-hand side
 */

template<int dim>
class RightHandSide : public Function<dim>
#if MESH_TYPE != 2
  ,
                      protected SolutionBase<dim>
#endif
{
public:
  RightHandSide() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const;
};

template<int dim>
double
RightHandSide<dim>::value(const Point<dim> & p, const unsigned int) const
{
  return 1.0;
}

/*
 *  Neumann boundary condition
 */

template<int dim>
class NeumannBoundary : public Function<dim>
{
public:
  NeumannBoundary(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  virtual ~NeumannBoundary(){};

  virtual double
  value(const Point<dim> & /* p */, const unsigned int /* component */) const
  {
    double result = 0.0;
    return result;
  }
};

/******************************************************************************/
/*                                                                            */
/*     GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR    */
/*                                                                            */
/******************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_conditions(
  std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
  unsigned int const                                n_refine_space,
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor,
  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> & /*periodic_faces*/)
{
  std::vector<std::string> files;
  get_lung_files_from_environment(files);
  auto tree_factory = dealii::GridGenerator::lung_files_to_node(files);

  std::map<std::string, double> timings;
  const int                     refinements = 4;

  // create triangulation
  if(auto tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
    dealii::GridGenerator::lung(
      *tria, refinements, n_refine_space, n_refine_space, tree_factory, timings);
  else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
    dealii::GridGenerator::lung(*tria, refinements, n_refine_space, tree_factory, timings);
  else
    AssertThrow(false, ExcMessage("Unknown triangulation!"));

  // set boundary conditions
  std::shared_ptr<Function<dim>> zero_function_scalar;
  zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));
  // boundary_descriptor->neumann_bc.insert({0, zero_function_scalar});
  boundary_descriptor->dirichlet_bc.insert({0, zero_function_scalar});
  boundary_descriptor->dirichlet_bc.insert({1, zero_function_scalar});
  boundary_descriptor->dirichlet_bc.insert({2, zero_function_scalar});
}

template<>
void create_grid_and_set_boundary_conditions<2>(
  std::shared_ptr<parallel::Triangulation<2>> /*triangulation*/,
  unsigned int const /*n_refine_space*/,
  std::shared_ptr<Poisson::BoundaryDescriptor<2>> /*boundary_descriptor*/,
  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<2>::cell_iterator>> & /*periodic_faces*/)
{
  AssertThrow(false, ExcMessage("The lung case can be only run in 3D!"));
}

template<int dim>
void
set_field_functions(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim>> analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());

  std::shared_ptr<Function<dim>> right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->analytical_solution = analytical_solution;
  field_functions->right_hand_side     = right_hand_side;
}

template<int dim>
void
set_analytical_solution(std::shared_ptr<Poisson::AnalyticalSolution<dim>> analytical_solution)
{
  analytical_solution->solution.reset(new AnalyticalSolution<dim>());
}
