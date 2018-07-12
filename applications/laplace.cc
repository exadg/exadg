/*
 * laplace.cc
 *
 *  Created on:
 *      Author:
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// laplace problems

#include "laplace_cases/cosinus.h"

using namespace dealii;

template <int dim, int fe_degree, typename Number = double>
class LaplaceProblem {
public:
  typedef double value_type;
  LaplaceProblem(const unsigned int n_refine_space);

  void solve_problem();

private:
  void print_header();

  void print_grid_data();

  void setup_postprocessor();

  ConditionalOStream pcout;
  parallel::distributed::Triangulation<dim> triangulation;
  const unsigned int n_refine_space;
  
};

template <int dim, int fe_degree, typename Number>
LaplaceProblem<dim, fe_degree, Number>::LaplaceProblem(
    const unsigned int n_refine_space_in)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      triangulation(MPI_COMM_WORLD, dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<
                        dim>::construct_multigrid_hierarchy),
      n_refine_space(n_refine_space_in) {
  print_header();
}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::print_header() {  
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                            Laplace/Poisson equation                             " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::print_grid_data() {}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::setup_postprocessor() {}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::solve_problem() {}

int main(int argc, char **argv) {
  try {
    // using namespace ConvectionDiffusionProblem;
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV
                << " on branch " << DEAL_II_GIT_BRANCH << std::endl
                << std::endl;
    }

    deallog.depth_console(0);

    // mesh refinements in order to perform spatial convergence tests
    for (unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
         refine_steps_space <= REFINE_STEPS_SPACE_MAX; ++refine_steps_space) {
      LaplaceProblem<DIMENSION, FE_DEGREE> conv_diff_problem(
          refine_steps_space);
      conv_diff_problem.solve_problem();
    }
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
