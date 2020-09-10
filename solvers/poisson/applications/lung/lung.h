/*
 * lung.h
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_POISSON_TEST_CASES_LUNG_H_
#define APPLICATIONS_POISSON_TEST_CASES_LUNG_H_

#include "../../../incompressible_flow_with_transport/applications/lung/grid/lung_environment.h"
#include "../../../incompressible_flow_with_transport/applications/lung/grid/lung_grid.h"

namespace ExaDG
{
namespace Poisson
{
namespace Lung
{
using namespace dealii;

// outlet boundary IDs
types::boundary_id const OUTLET_ID_FIRST = 2;
types::boundary_id       OUTLET_ID_LAST  = OUTLET_ID_FIRST; // initialization

enum class BoundaryConditionType
{
  Dirichlet,
  DirichletNeumann
};

BoundaryConditionType const BOUNDARY_CONDITION_TYPE = BoundaryConditionType::DirichletNeumann;

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC() : Function<dim>(1, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(100.0 * p[d]);

    return result;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("DirectoryLungFiles", directory_lung_files, "Directory where to find files for lung geometry.");
      prm.add_parameter("Generations",        n_generations,        "Number of generations.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string  directory_lung_files = "";
  unsigned int n_generations        = 4;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    param.triangulation_type     = TriangulationType::Distributed;
    param.mapping                = MappingType::Affine; // Isoparametric;
    param.spatial_discretization = SpatialDiscretization::DG;
    param.IP_factor              = 1.0;

    // SOLVER
    param.solver                      = Poisson::Solver::CG;
    param.solver_data.abs_tol         = 1.e-20;
    param.solver_data.rel_tol         = 1.e-10;
    param.solver_data.max_iter        = 1e4;
    param.solver_data.max_krylov_size = 200;
    param.compute_performance_metrics = true;
    param.preconditioner              = Preconditioner::Multigrid;
    param.multigrid_data.type         = MultigridType::cphMG;
    param.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    param.multigrid_data.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    param.multigrid_data.smoother_data.iterations = 5;
    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data.coarse_problem.solver_data.rel_tol           = 1.e-1;
    param.multigrid_data.coarse_problem.amg_data.data.smoother_type   = "ILU"; //"Chebyshev";
    param.multigrid_data.coarse_problem.amg_data.data.smoother_sweeps = 1;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim = 3!"));

    std::vector<std::string> files;
    files.push_back(directory_lung_files + "leftbot.dat");
    files.push_back(directory_lung_files + "lefttop.dat");
    files.push_back(directory_lung_files + "rightbot.dat");
    files.push_back(directory_lung_files + "rightmid.dat");
    files.push_back(directory_lung_files + "righttop.dat");
    auto tree_factory = ExaDG::GridGen::lung_files_to_node(files);

    std::string spline_file = directory_lung_files + "../splines_raw6.dat";

    std::map<std::string, double> timings;

    std::shared_ptr<LungID::Checker> generation_limiter(
      new LungID::GenerationChecker(n_generations));
    // std::shared_ptr<LungID::Checker> generation_limiter(new LungID::ManualChecker());

    // create triangulation
    if(auto tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
    {
      ExaDG::GridGen::lung(*tria,
                           n_refine_space,
                           n_refine_space,
                           tree_factory,
                           timings,
                           OUTLET_ID_FIRST,
                           OUTLET_ID_LAST,
                           spline_file,
                           generation_limiter);
    }
    else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
    {
      ExaDG::GridGen::lung(*tria,
                           n_refine_space,
                           tree_factory,
                           timings,
                           OUTLET_ID_FIRST,
                           OUTLET_ID_LAST,
                           spline_file,
                           generation_limiter);
    }
    else
    {
      AssertThrow(false, ExcMessage("Unknown triangulation!"));
    }

    AssertThrow(OUTLET_ID_LAST - OUTLET_ID_FIRST == std::pow(2, n_generations - 1),
                ExcMessage("Number of outlets has to be 2^{N_generations-1}."));
  }

  void set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    if(BOUNDARY_CONDITION_TYPE == BoundaryConditionType::DirichletNeumann)
    {
      // 0 = walls -> Neumann
      boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));

      // 1 = inlet -> Dirichlet with constant value of 1
      boundary_descriptor->dirichlet_bc.insert(
        pair(1, new Functions::ConstantFunction<dim>(1 /* value */, 1 /* components */)));

      // all outlets -> homogeneous Dirichlet boundary conditions
      for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
      {
        boundary_descriptor->dirichlet_bc.insert(pair(id, new Functions::ZeroFunction<dim>(1)));
      }
    }
    else if(BOUNDARY_CONDITION_TYPE == BoundaryConditionType::Dirichlet)
    {
      // 0 = walls
      boundary_descriptor->dirichlet_bc.insert(pair(0, new DirichletBC<dim>()));

      // 1 = inlet
      boundary_descriptor->dirichlet_bc.insert(pair(1, new DirichletBC<dim>()));

      // all outlets
      for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
      {
        boundary_descriptor->dirichlet_bc.insert(pair(id, new DirichletBC<dim>()));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.output_folder      = this->output_directory + "vtu/";
    pp_data.output_data.output_name        = this->output_name;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = degree;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Lung
} // namespace Poisson
} // namespace ExaDG

#endif
