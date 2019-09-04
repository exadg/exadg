/*
 * poisson.cc
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// spatial discretization
#include "../include/poisson/spatial_discretization/operator.h"

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor_base.h"

// user interface, etc.
#include "poisson/user_interface/analytical_solution.h"
#include "poisson/user_interface/boundary_descriptor.h"
#include "poisson/user_interface/field_functions.h"
#include "poisson/user_interface/input_parameters.h"

// functionalities
#include "functionalities/calculate_maximum_aspect_ratio.h"
#include "functionalities/mesh_resolution_generator_hypercube.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"


// specify the test case that has to be solved

// template
#include "poisson_test_cases/template.h"

//#include "poisson_test_cases/gaussian.h"
//#include "poisson_test_cases/slit.h"
//#include "poisson_test_cases/sine.h"
//#include "poisson_test_cases/nozzle.h"
//#include "poisson_test_cases/torus.h"
//#include "poisson_test_cases/lung.h"

using namespace dealii;
using namespace Poisson;

RunType const RUN_TYPE = RunType::RefineHAndP; // FixedProblemSize //IncreasingProblemSize

/*
 * Specify minimum and maximum problem size for
 *  RunType::FixedProblemSize
 *  RunType::IncreasingProblemSize
 */
types::global_dof_index N_DOFS_MIN = 2.5e5;
types::global_dof_index N_DOFS_MAX = 7.5e5;

/*
 * Enable hyper_cube meshes with number of cells per direction other than multiples of 2.
 * Use this only for simple hyper_cube problems and for
 *  RunType::FixedProblemSize
 *  RunType::IncreasingProblemSize
 */
//#define ENABLE_SUBDIVIDED_HYPERCUBE

#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
// will be set automatically for RunType::FixedProblemSize and RunType::IncreasingProblemSize
unsigned int SUBDIVISIONS_MESH = 1;
#endif

struct Timings
{
  Timings() : degree(1), dofs(1), n_10(0), tau_10(0.0)
  {
  }

  Timings(unsigned int const            degree_,
          types::global_dof_index const dofs_,
          double const                  n_10_,
          double const                  tau_10_)
    : degree(degree_), dofs(dofs_), n_10(n_10_), tau_10(tau_10_)
  {
  }

  void
  print_header(ConditionalOStream const & pcout) const
  {
    // names
    pcout << std::setw(7) << "degree";
    pcout << std::setw(15) << "dofs";
    pcout << std::setw(8) << "n_10";
    pcout << std::setw(15) << "tau_10";
    pcout << std::setw(15) << "throughput";
    pcout << std::endl;

    // units
    pcout << std::setw(7) << " ";
    pcout << std::setw(15) << " ";
    pcout << std::setw(8) << " ";
    pcout << std::setw(15) << "in s*core/DoF";
    pcout << std::setw(15) << "in DoF/s/core";
    pcout << std::endl;

    pcout << std::endl;
  }

  void
  print_results(ConditionalOStream const & pcout) const
  {
    pcout << std::setw(7) << std::fixed << degree;
    pcout << std::setw(15) << std::fixed << dofs;
    pcout << std::setw(8) << std::fixed << std::setprecision(1) << n_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << tau_10;
    pcout << std::setw(15) << std::scientific << std::setprecision(2) << 1.0 / tau_10;
    pcout << std::endl;
  }

  unsigned int            degree;
  types::global_dof_index dofs;
  double                  n_10;
  double                  tau_10;
};
// global variable used to store the wall times for different polynomial degrees and problem sizes
std::vector<Timings> timings;

class ProblemBase
{
public:
  virtual ~ProblemBase()
  {
  }

  virtual void
  setup(InputParameters const & param) = 0;

  virtual void
  solve() = 0;

  virtual void
  analyze_computing_times() const = 0;
};

template<int dim, typename Number = double>
class Problem : public ProblemBase
{
public:
  Problem();

  void
  setup(InputParameters const & param);

  void
  solve();

  void
  analyze_computing_times() const;

private:
  void
  print_header();

  ConditionalOStream pcout;

  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<DGOperator<dim, Number>> poisson_operator;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  // number of iterations
  mutable unsigned int iterations;
  mutable double wall_time_vector_init, wall_time_rhs, wall_time_solver, wall_time_postprocessing;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem()
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    overall_time(0.0),
    setup_time(0.0),
    iterations(0),
    wall_time_vector_init(0.0),
    wall_time_rhs(0.0),
    wall_time_solver(0.0),
    wall_time_postprocessing(0.0)
{
}

template<int dim, typename Number>
void
Problem<dim, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                            scalar Poisson equation                              " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Problem<dim, Number>::setup(InputParameters const & param_in)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout);

  param = param_in;
  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  // triangulation
  if(param.triangulation_type == TriangulationType::Distributed)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
  create_grid_and_set_boundary_ids(triangulation,
                                   param.h_refinements,
                                   periodic_faces,
                                   SUBDIVISIONS_MESH);
#else
  create_grid_and_set_boundary_ids(triangulation, param.h_refinements, periodic_faces);
#endif
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  set_boundary_conditions(boundary_descriptor);

  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, Number>(param);

  // initialize Poisson operator
  poisson_operator.reset(new DGOperator<dim, Number>(*triangulation, param, postprocessor));

  poisson_operator->setup(periodic_faces, boundary_descriptor, field_functions);

  if(false)
  {
    double AR = calculate_aspect_ratio_vertex_distance(*triangulation);
    std::cout << std::endl << "Maximum aspect ratio vertex distance = " << AR << std::endl;
    AR = poisson_operator->calculate_maximum_aspect_ratio();
    std::cout << std::endl << "Maximum aspect ratio Jacobian = " << AR << std::endl;
  }

  poisson_operator->setup_solver();

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve()
{
  Timer timer_local;

  // initialization of vectors
  timer_local.restart();
  LinearAlgebra::distributed::Vector<Number> rhs;
  LinearAlgebra::distributed::Vector<Number> sol;
  poisson_operator->initialize_dof_vector(rhs);
  poisson_operator->initialize_dof_vector(sol);
  poisson_operator->prescribe_initial_conditions(sol);
  wall_time_vector_init = timer_local.wall_time();

  // postprocessing of results
  timer_local.restart();
  poisson_operator->do_postprocessing(sol);
  wall_time_postprocessing = timer_local.wall_time();

  // calculate right-hand side
  timer_local.restart();
  poisson_operator->rhs(rhs);
  wall_time_rhs = timer_local.wall_time();

  // solve linear system of equations
  timer_local.restart();
  iterations       = poisson_operator->solve(sol, rhs);
  wall_time_solver = timer_local.wall_time();

  // postprocessing of results
  timer_local.restart();
  poisson_operator->do_postprocessing(sol);
  wall_time_postprocessing += timer_local.wall_time();

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for Poisson solver:" << std::endl;

  double const n_10 = poisson_operator->get_n10();
  // Iterations
  {
    this->pcout << std::endl << "Number of iterations:" << std::endl;

    this->pcout << "  Iterations n         = " << std::fixed << iterations << std::endl;

    this->pcout << "  Iterations n_10      = " << std::fixed << std::setprecision(1) << n_10
                << std::endl;

    this->pcout << "  Convergence rate rho = " << std::fixed << std::setprecision(4)
                << poisson_operator->get_average_convergence_rate() << std::endl;
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names = {"Initialization of vectors",
                                    "Right-hand side",
                                    "Linear solver",
                                    "Postprocessing"};

  std::vector<double> computing_times;
  computing_times.resize(4);
  computing_times[0] = wall_time_vector_init;
  computing_times[1] = wall_time_rhs;
  computing_times[2] = wall_time_solver;
  computing_times[3] = wall_time_postprocessing;

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data =
    Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD);
  double const setup_time_avg = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - sum_of_substeps - setup_time_avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  // Throughput in DoF/s per time step per core
  unsigned int                  N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  types::global_dof_index const DoFs            = poisson_operator->get_number_of_dofs();

  this->pcout << std::endl
              << "Computational costs and throughput:" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Degrees of freedom      = " << DoFs << std::endl
              << std::endl;

  this->pcout << "Overall costs (including setup + postprocessing):" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (overall_time_avg * N_mpi_processes) << " DoF/s/core" << std::endl
              << std::endl;

  this->pcout << "Linear solver:" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << wall_time_solver << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << wall_time_solver * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (wall_time_solver * N_mpi_processes) << " DoF/s/core" << std::endl
              << std::endl;

  double const t_10   = wall_time_solver * n_10 / iterations;
  double const tau_10 = t_10 * (double)N_mpi_processes / DoFs;
  this->pcout << "Linear solver (numbers based on n_10):" << std::endl
              << "  Wall time t_10          = " << std::scientific << std::setprecision(2) << t_10
              << " s" << std::endl
              << "  tau_10                  = " << std::scientific << std::setprecision(2) << tau_10
              << " s*core/DoF" << std::endl
              << "  Throughput E_10         = " << std::scientific << std::setprecision(2)
              << 1.0 / tau_10 << " DoF/s/core" << std::endl;

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  timings.push_back(Timings(param.degree, DoFs, n_10, tau_10));
}

void
do_run(InputParameters const & param)
{
  // setup problem and run simulation
  typedef double               Number;
  std::shared_ptr<ProblemBase> problem;

  if(param.dim == 2)
    problem.reset(new Problem<2, Number>());
  else if(param.dim == 3)
    problem.reset(new Problem<3, Number>());
  else
    AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

  problem->setup(param);

  try
  {
    problem->solve();
  }
  catch(...)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Solver failed to converge!" << std::endl
                << "----------------------------------------------------" << std::endl;
    }
  }

  problem->analyze_computing_times();
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    // set parameters
    InputParameters param;
    set_input_parameters(param);

    if(RUN_TYPE == RunType::RefineHAndP)
    {
      // p-refinement
      for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
      {
        // reset degree
        param.degree = degree;

        // h-refinement
        for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
            ++h_refinements)
        {
          // reset mesh refinement
          param.h_refinements = h_refinements;

          do_run(param);
        }
      }
    }
#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
    else if(RUN_TYPE == RunType::FixedProblemSize || RUN_TYPE == RunType::IncreasingProblemSize)
    {
      // a vector storing tuples of the form (degree k, refine level l, n_subdivisions_1d)
      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> resolutions;

      // fill resolutions vector

      if(RUN_TYPE == RunType::IncreasingProblemSize)
      {
        AssertThrow(
          DEGREE_MIN == DEGREE_MAX,
          ExcMessage(
            "Only a single polynomial degree can be considered for RunType::IncreasingProblemSize"));
      }

      // k-refinement
      for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
      {
        unsigned int const dim              = double(param.dim);
        double const       dofs_per_element = std::pow(degree + 1, dim);

        fill_resolutions_vector(
          resolutions, degree, dim, dofs_per_element, N_DOFS_MIN, N_DOFS_MAX, RUN_TYPE);
      }

      // loop over resolutions vector and run simulations
      for(auto iter = resolutions.begin(); iter != resolutions.end(); ++iter)
      {
        param.degree        = std::get<0>(*iter);
        param.h_refinements = std::get<1>(*iter);
        SUBDIVISIONS_MESH   = std::get<2>(*iter);

        do_run(param);
      }
    }
#endif
    else
    {
      AssertThrow(false,
                  ExcMessage("Not implemented. Make sure to activate ENABLE_SUBDIVIDED_HYPERCUBE "
                             "for RunType::FixedProblemSize or RunType::IncreasingProblemSize."));
    }

    // summarize results for all polynomial degrees and problem sizes
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "_________________________________________________________________________________"
          << std::endl
          << std::endl;

    pcout << "Summary of performance results for Poisson solver:" << std::endl << std::endl;

    timings[0].print_header(pcout);
    for(std::vector<Timings>::const_iterator it = timings.begin(); it != timings.end(); ++it)
      it->print_results(pcout);

    pcout << "_________________________________________________________________________________"
          << std::endl
          << std::endl;
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  return 0;
}
