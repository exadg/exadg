/*
 * poisson.cc
 *
 *  Created on: 2018
 *      Author: münch
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <vector>

#include "../include/poisson/spatial_discretization/laplace_operator.h"
#include "../include/poisson/spatial_discretization/poisson_operation.h"

#include "../include/poisson/user_interface/analytical_solution.h"
#include "../include/poisson/user_interface/boundary_descriptor.h"
#include "../include/poisson/user_interface/field_functions.h"
#include "../include/poisson/user_interface/input_parameters.h"

#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// laplace problems
//#include "poisson_test_cases/cosinus.h"
#include "poisson_test_cases/torus.h"

using namespace dealii;
using namespace Poisson;

const int best_of = 1;

template<int dim, int fe_degree, typename Function>
void
measure_minimum_time(const unsigned int best_of,
                     ConvergenceTable & table,
                     std::string        label,
                     Function           f)
{
  Timer  timer;
  double min_time = std::numeric_limits<double>::max();
  for(unsigned int i = 0; i < best_of; i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    timer.restart();
    f();
    double temp = timer.wall_time();
    min_time    = std::min(min_time, temp);
  }
  table.add_value(label, min_time);
  table.set_scientific(label, true);
}

template<int dim, int fe_degree, typename Number = double>
class PoissonProblem
{
public:
  typedef double value_type;
  PoissonProblem(const unsigned int n_refine_space);

  void
  solve_problem(ConvergenceTable & convergence_table);

private:
  void
  print_header();

  void
  print_grid_data();

  template<typename Vec>
  void
  output_data(std::string filename, Vec & solution)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(poisson_operation->get_dof_handler());
    data_out.add_data_vector(solution, "solution");

    auto ranks = solution;
    int  rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ranks = rank;
    data_out.add_data_vector(ranks, "ranks");

    data_out.build_patches(fe_degree + 1);

    data_out.write_vtu_in_parallel(filename.c_str(), MPI_COMM_WORLD);
  }

  ConditionalOStream                        pcout;
  parallel::distributed::Triangulation<dim> triangulation;
  const unsigned int                        n_refine_space;
  Poisson::InputParameters                  param;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<Poisson::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<Poisson::AnalyticalSolution<dim>> analytical_solution;

  std::shared_ptr<Poisson::DGOperation<dim, fe_degree, value_type>> poisson_operation;
};

template<int dim, int fe_degree, typename Number>
PoissonProblem<dim, fe_degree, Number>::PoissonProblem(const unsigned int n_refine_space_in)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space(n_refine_space_in)
{
  print_header();
  param.set_input_parameters();
  param.check_input_parameters();

  print_MPI_info(pcout);
  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new Poisson::FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new Poisson::AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor.reset(new Poisson::BoundaryDescriptor<dim>());

  poisson_operation.reset(new Poisson::DGOperation<dim, fe_degree, Number>(triangulation, param));
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
        << "_________________________________________________________________________________" << std::endl
        << "                                                                                 " << std::endl
        << "                High-order discontinuous Galerkin solver for the                 " << std::endl
        << "                                Poisson equation                                 " << std::endl
        << "_________________________________________________________________________________" << std::endl
        << std::endl;
  // clang-format on
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::print_grid_data()
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout, "Number of refinements", n_refine_space);
  print_parameter(pcout, "Number of cells", triangulation.n_global_active_cells());
  print_parameter(pcout, "Number of faces", triangulation.n_active_faces());
  print_parameter(pcout, "Number of vertices", triangulation.n_vertices());
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::solve_problem(ConvergenceTable & convergence_table)
{
  // create grid and set bc
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor,
                                          periodic_faces);

  print_grid_data();

  // setup poisson operation
  poisson_operation->setup(periodic_faces, boundary_descriptor, field_functions);

  Timer timer;
  poisson_operation->setup_solver();
  double time_setup = timer.wall_time();

  // allocate vectors
  parallel::distributed::Vector<Number> rhs;
  parallel::distributed::Vector<Number> solution;
  poisson_operation->initialize_dof_vector(rhs);
  poisson_operation->initialize_dof_vector(solution);

  // solve problem
  convergence_table.add_value("dim", dim);
  convergence_table.add_value("degree", fe_degree);
  convergence_table.add_value("refs", n_refine_space);
  convergence_table.add_value("dofs", solution.size());
  convergence_table.add_value("setup", time_setup);
  convergence_table.set_scientific("setup", true);

  if(param.output_data.write_output)
    this->output_data(param.output_data.output_folder + param.output_data.output_name + "0.vtu",
                      solution);

  measure_minimum_time<dim, fe_degree>(best_of, convergence_table, "rhs", [&]() mutable {
    poisson_operation->rhs(rhs);
  });

  int cycles;
  measure_minimum_time<dim, fe_degree>(best_of, convergence_table, "solve", [&]() mutable {
    cycles = poisson_operation->solve(solution, rhs);
  });

  convergence_table.add_value("cycles", cycles);

  if(param.output_data.write_output)
    this->output_data(param.output_data.output_folder + param.output_data.output_name + "1.vtu",
                      solution);
}

template<int dim, int fe_degree>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table)
  {
    std::vector<int> sizes = {/*32,*/ 64 /*,128,256,512,1024,2048,4096,8192*/};

    for(auto size : sizes)
    {
      if(size > 5000 && fe_degree <= 5)
        continue;
      int refinement = std::log(size / fe_degree) / std::log(2.0);

      PoissonProblem<dim, fe_degree> conv_diff_problem(refinement);
      conv_diff_problem.solve_problem(convergence_table);
    }
  }
};

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if(!rank)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl
                << std::endl;
    }

    deallog.depth_console(0);

    ConvergenceTable convergence_table;

    Run<DIMENSION,2>::run(convergence_table);
//    Run<DIMENSION,3>::run(convergence_table);
//    Run<DIMENSION,4>::run(convergence_table);
//    Run<DIMENSION,5>::run(convergence_table);
//    Run<DIMENSION,6>::run(convergence_table);
//    Run<DIMENSION,7>::run(convergence_table);
//    Run<DIMENSION,8>::run(convergence_table);
//    Run<DIMENSION,9>::run(convergence_table);

    if(!rank)
    {
      std::ofstream outfile;
      outfile.open("ctable.csv");
      convergence_table.write_text(std::cout);
      convergence_table.write_text(outfile);
      outfile.close();
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
