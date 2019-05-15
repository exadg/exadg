/*
 * convection_diffusion.cc
 *
 *  Created on: Aug 18, 2016
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
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"
#include "poisson/user_interface/analytical_solution.h"
#include "poisson/user_interface/boundary_descriptor.h"
#include "poisson/user_interface/field_functions.h"
#include "poisson/user_interface/input_parameters.h"


// specify the test case that has to be solved

// template
#include "poisson_test_cases/template.h"

//#include "poisson_test_cases/gaussian.h"
//#include "poisson_test_cases/cosinus.h"
//#include "poisson_test_cases/torus.h"
//#include "poisson_test_cases/lung.h"

using namespace dealii;
using namespace Poisson;

template<typename Number>
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
class Problem : public ProblemBase<Number>
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

  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<Poisson::AnalyticalSolution<dim>> analytical_solution;

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
  print_dealii_info(pcout);
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

  create_grid_and_set_boundary_ids(triangulation, param.h_refinements, periodic_faces);
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  set_boundary_conditions(boundary_descriptor);

  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, Number>();

  // initialize Poisson operator
  poisson_operator.reset(new DGOperator<dim, Number>(*triangulation, param, postprocessor));

  poisson_operator->setup(periodic_faces,
                          boundary_descriptor,
                          field_functions,
                          analytical_solution);

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

  // Iterations
  {
    this->pcout << std::endl << "Number of iterations:" << std::endl;

    this->pcout << "  Iterations: " << std::fixed << std::setprecision(2) << std::right
                << std::setw(6) << iterations << std::endl;
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
  // Throughput in DoFs/s per time step per core
  unsigned int                  N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  types::global_dof_index const DoFs            = poisson_operator->get_number_of_dofs();

  this->pcout << std::endl
              << "Computational costs and throughput:" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Degrees of freedom      = " << DoFs << std::endl
              << std::endl
              << "Overall costs (including setup + postprocessing):" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (overall_time_avg * N_mpi_processes) << " DoFs/s/core" << std::endl
              << std::endl
              << "Right-hand side + solver:" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << wall_time_rhs + wall_time_solver << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << (wall_time_rhs + wall_time_solver) * (double)N_mpi_processes / 3600.0 << " CPUh"
              << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / ((wall_time_rhs + wall_time_solver) * N_mpi_processes) << " DoFs/s/core"
              << std::endl;

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}


// instantiations
template class Problem<2, double>;
template class Problem<3, double>;

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    // set parameters
    Poisson::InputParameters param;
    set_input_parameters(param);

    // k-refinement
    for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
    {
      // h-refinement
      for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
          ++h_refinements)
      {
        // reset degree
        param.degree = degree;

        // reset mesh refinement
        param.h_refinements = h_refinements;

        // setup problem and run simulation
        typedef double                       Number;
        std::shared_ptr<ProblemBase<Number>> problem;

        if(param.dim == 2)
          problem.reset(new Problem<2, Number>());
        else if(param.dim == 3)
          problem.reset(new Problem<3, Number>());
        else
          AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

        problem->setup(param);

        problem->solve();

        problem->analyze_computing_times();
      }
    }
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
