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

#include "../include/laplace/spatial_discretization/laplace_operator.h"
#include "../include/laplace/spatial_discretization/poisson_operation.h"

#include "../include/laplace/user_interface/analytical_solution.h"
#include "../include/laplace/user_interface/boundary_descriptor.h"
#include "../include/laplace/user_interface/field_functions.h"
#include "../include/laplace/user_interface/input_parameters.h"

#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// laplace problems
#include "laplace_cases/cosinus.h"

using namespace dealii;
using namespace Laplace;

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
  Laplace::InputParameters param;
  
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;
  
  std::shared_ptr<Laplace::FieldFunctions<dim> > field_functions;
  std::shared_ptr<Laplace::BoundaryDescriptor<dim> > boundary_descriptor;
  std::shared_ptr<Laplace::AnalyticalSolution<dim> > analytical_solution;
  
  std::shared_ptr<Laplace::DGOperation<dim,fe_degree, value_type> > poisson_operation;
  
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
  param.set_input_parameters();
  param.check_input_parameters();
  
  print_MPI_info(pcout);
  if(param.print_input_parameters == true)
    param.print(pcout);
  
  field_functions.reset(new Laplace::FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new Laplace::AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor.reset(new Laplace::BoundaryDescriptor<dim>());

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
void LaplaceProblem<dim, fe_degree, Number>::print_grid_data() {
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout,"Number of refinements",n_refine_space);
  print_parameter(pcout,"Number of cells",triangulation.n_global_active_cells());
  print_parameter(pcout,"Number of faces",triangulation.n_active_faces());
  print_parameter(pcout,"Number of vertices",triangulation.n_vertices());
}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::setup_postprocessor() {}

template <int dim, int fe_degree, typename Number>
void LaplaceProblem<dim, fe_degree, Number>::solve_problem() {
  // create grid and set bc
  create_grid_and_set_boundary_conditions(triangulation, n_refine_space, boundary_descriptor);
  print_grid_data();

  // setup poisson operation
  poisson_operation->setup(periodic_faces, boundary_descriptor, field_functions);
  poisson_operation->setup_solver();
  
  // setup postprocessor
  setup_postprocessor();
  
  // compute right hand side

  // solve problem
  //poisson_operation->solve_problem();
}

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
