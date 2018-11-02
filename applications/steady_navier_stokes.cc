/*
 * steady_navier_stokes.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/driver_steady_problems.h"

// Paramters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/stokes_curl_flow.h"
//#include "incompressible_navier_stokes_test_cases/couette.h"
#include "incompressible_navier_stokes_test_cases/poiseuille.h"
//#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
//#include "incompressible_navier_stokes_test_cases/flow_past_cylinder.h"


template<int dim, int degree_u, int degree_p = degree_u - 1, typename Number = double>
class NavierStokesProblem
{
public:
  NavierStokesProblem(const unsigned int refine_steps_space,
                      const unsigned int refine_steps_time = 0);

  void
  solve_problem();

private:
  void
  print_header();

  void
  setup_navier_stokes_operation();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  const unsigned int n_refine_space;

  std::shared_ptr<FieldFunctions<dim>>      field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolution<dim>> analytical_solution;

  InputParameters<dim> param;

  typedef DGNavierStokesCoupled<dim, degree_u, degree_p, Number> DGCoupled;

  std::shared_ptr<DGCoupled> navier_stokes_operation;

  typedef PostProcessorBase<dim, degree_u, degree_p, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  typedef DriverSteadyProblems<dim, Number, DGCoupled> DriverSteady;

  std::shared_ptr<DriverSteady> driver_steady;
};

template<int dim, int degree_u, int degree_p, typename Number>
NavierStokesProblem<dim, degree_u, degree_p, Number>::NavierStokesProblem(
  unsigned int const refine_steps_space,
  unsigned int const /*refine_steps_time*/)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space(refine_steps_space)
{
  param.set_input_parameters();
  param.check_input_parameters();

  print_header();
  print_MPI_info(pcout);
  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new FieldFunctions<dim>());

  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolution<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_analytical_solution(analytical_solution);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  AssertThrow(param.solver_type == SolverType::Steady,
              ExcMessage("This is a steady solver. Check input parameters."));

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, degree_u, degree_p, Number>(param);

  // initialize navier_stokes_operation
  navier_stokes_operation.reset(new DGCoupled(triangulation, param, postprocessor));

  // initialize driver for steady state problem that depends on navier_stokes_operation
  driver_steady.reset(new DriverSteady(navier_stokes_operation, param));
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                 steady, incompressible Navier-Stokes equations                  " << std::endl
  << "            based on coupled solution approach of Newton-Krylov type             " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_navier_stokes_operation()
{
  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions,
                                 analytical_solution);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::solve_problem()
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure,
                                          periodic_faces);

  print_grid_data(pcout, n_refine_space, triangulation);

  setup_navier_stokes_operation();

  driver_steady->setup();

  navier_stokes_operation->setup_solvers();

  driver_steady->solve_steady_problem();
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl
                << std::endl;
    }

    deallog.depth_console(0);

    // mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX;
        ++refine_steps_space)
    {
      // time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;
          refine_steps_time <= REFINE_STEPS_TIME_MAX;
          ++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, VALUE_TYPE>
          navier_stokes_problem(refine_steps_space, refine_steps_time);

        navier_stokes_problem.solve_problem();
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
