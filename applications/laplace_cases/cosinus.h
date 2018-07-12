#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/******************************************************************************/
/*                                                                            */
/*                             INPUT PARAMETERS                               */
/*                                                                            */
/******************************************************************************/

const unsigned int DIMENSION = 2;
const unsigned int FE_DEGREE = 7;
const unsigned int REFINE_STEPS_SPACE_MIN = 5;
const unsigned int REFINE_STEPS_SPACE_MAX = 5;

void Laplace::InputParameters::set_input_parameters()
{
  // MATHEMATICAL MODEL
  right_hand_side = true;

  // PHYSICAL QUANTITIES

  // TEMPORAL DISCRETIZATION

  // SPATIAL DISCRETIZATION
  IP_factor = 1.0;

  // SOLVER
  solver = Solver::PCG;
  abs_tol = 1.e-20;
  rel_tol = 1.e-8;
  max_iter = 1e4;
  preconditioner = Preconditioner::Multigrid;
  // MG smoother
  multigrid_data.smoother = MultigridSmoother::Chebyshev;
  // MG smoother data
  multigrid_data.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::None;
  multigrid_data.gmres_smoother_data.number_of_iterations = 5;
  // MG coarse grid solver
  multigrid_data.coarse_solver = MultigridCoarseGridSolver::GMRES_PointJacobi;
  
  multigrid_data.coarse_solver = MultigridCoarseGridSolver::AMG_ML;
  //multigrid_data.two_levels = true;
  multigrid_data.type = MultigridType::PGMG;
  
}

/******************************************************************************/
/*                                                                            */
/* FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.) */
/*                                                                            */ 
/*                                                                            */
/******************************************************************************/

/*
 *  Analytical solution
 */

template <int dim> class AnalyticalSolution : public Function<dim> {
public:
  AnalyticalSolution(const unsigned int n_components = 1,
                     const double time = 0.)
      : Function<dim>(n_components, time) {}

  virtual ~AnalyticalSolution(){};

  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int /*component*/ = 0) const {
    double result = 0.0;
    return result;
  }
};

/*
 *  Right-hand side
 */

template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}

  virtual ~RightHandSide(){};

  virtual double value(const Point<dim> & p,
                       const unsigned int /* component */) const {
    const double coef = 1.0;
    double temp=1;
    for(int i = 0; i < dim; i++) 
        temp *=std::cos(p[i]);
    return temp * dim * coef;
  }
};

/*
 *  Neumann boundary condition
 */

template <int dim> class NeumannBoundary : public Function<dim> {
public:
  NeumannBoundary(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}

  virtual ~NeumannBoundary(){};

  virtual double value(const Point<dim> & /* p */,
                       const unsigned int /* component */) const {
    double result = 0.0;
    return result;
  }
};

/******************************************************************************/
/*                                                                            */
/*     GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR    */
/*                                                                            */
/******************************************************************************/

template <int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim> &triangulation,
    unsigned int const n_refine_space,
    std::shared_ptr<Laplace::BoundaryDescriptor<dim>> boundary_descriptor) {
  // hypercube: [left,right]^dim
  const double left = -0.5*numbers::PI, right = +0.5*numbers::PI;
  GridGenerator::hyper_cube(triangulation, left, right);

  triangulation.refine_global(n_refine_space);

  // dirichlet bc:
  std::shared_ptr<Function<dim>> analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());
  boundary_descriptor->dirichlet_bc.insert({0, analytical_solution});

  // neumann bc:
  std::shared_ptr<Function<dim>> neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>());
  boundary_descriptor->neumann_bc.insert({1, neumann_bc});
}

template <int dim>
void set_field_functions(
    std::shared_ptr<Laplace::FieldFunctions<dim>> field_functions) {
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim>> analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>());

  std::shared_ptr<Function<dim>> right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->analytical_solution = analytical_solution;
  field_functions->right_hand_side = right_hand_side;
}

template <int dim>
void set_analytical_solution(
    std::shared_ptr<Laplace::AnalyticalSolution<dim>> analytical_solution) {
  analytical_solution->solution.reset(new AnalyticalSolution<dim>(1));
}