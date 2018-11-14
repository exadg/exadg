/*
 * unsteady_compressible_navier_stokes.cc
 *
 *  Created on: 2018
 *      Author: fehn
 */


// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// postprocessor
#include "../include/compressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/compressible_navier_stokes/spatial_discretization/dg_comp_navier_stokes.h"

// temporal discretization
#include "../include/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.h"

// Paramters, BCs, etc.
#include "../include/compressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/compressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/compressible_navier_stokes/user_interface/field_functions.h"
#include "../include/compressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// Euler equations
//#include "compressible_navier_stokes_test_cases/euler_vortex_flow.h"

// Navier-Stokes equations
//#include "compressible_navier_stokes_test_cases/channel_flow.h"
//#include "compressible_navier_stokes_test_cases/couette_flow.h"
//#include "compressible_navier_stokes_test_cases/steady_shear_flow.h"
//#include "compressible_navier_stokes_test_cases/manufactured_solution.h"
//#include "compressible_navier_stokes_test_cases/flow_past_cylinder.h"
#include "compressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"
//#include "compressible_navier_stokes_test_cases/turbulent_channel.h"

using namespace dealii;
using namespace CompNS;

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis>
class CompressibleNavierStokesProblem
{
public:
  typedef double value_type;

  typedef DGCompNavierStokesOperation<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type>
    DG_OPERATOR;

  typedef TimeIntExplRKCompNavierStokes<dim, fe_degree, value_type, DG_OPERATOR> TIME_INT;

  CompressibleNavierStokesProblem(const unsigned int refine_steps_space,
                                  const unsigned int refine_steps_time = 0);

  void
  solve_problem();

private:
  void
  print_header();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  const unsigned int n_refine_space;
  const unsigned int n_refine_time;

  std::shared_ptr<CompNS::FieldFunctions<dim>>           field_functions;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;
  std::shared_ptr<CompNS::AnalyticalSolution<dim>>       analytical_solution;

  CompNS::InputParameters<dim> param;

  std::shared_ptr<DG_OPERATOR> comp_navier_stokes_operator;

  std::shared_ptr<
    CompNS::PostProcessor<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type>>
    postprocessor;

  std::shared_ptr<TIME_INT> time_integrator;
};

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis>
CompressibleNavierStokesProblem<dim, fe_degree, n_q_points_conv, n_q_points_vis>::
  CompressibleNavierStokesProblem(const unsigned int n_refine_space_in,
                                  const unsigned int n_refine_time_in)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space(n_refine_space_in),
    n_refine_time(n_refine_time_in)
{
  print_header();
  print_MPI_info(pcout);

  param.set_input_parameters();
  param.check_input_parameters();

  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new CompNS::FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new CompNS::AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor_density.reset(new CompNS::BoundaryDescriptor<dim>());
  boundary_descriptor_velocity.reset(new CompNS::BoundaryDescriptor<dim>());
  boundary_descriptor_pressure.reset(new CompNS::BoundaryDescriptor<dim>());
  boundary_descriptor_energy.reset(new CompNS::BoundaryDescriptorEnergy<dim>());

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor =
    construct_postprocessor<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type>(param);

  // initialize compressible Navier-Stokes operator
  comp_navier_stokes_operator.reset(new DG_OPERATOR(triangulation, param, postprocessor));

  // initialize time integrator
  time_integrator.reset(new TIME_INT(comp_navier_stokes_operator, param, n_refine_time));
}

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis>
void
CompressibleNavierStokesProblem<dim, fe_degree, n_q_points_conv, n_q_points_vis>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                 unsteady, compressible Navier-Stokes equations                  " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis>
void
CompressibleNavierStokesProblem<dim, fe_degree, n_q_points_conv, n_q_points_vis>::solve_problem()
{
  // this function has to be defined in the header file that implements
  // all problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_density,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure,
                                          boundary_descriptor_energy,
                                          periodic_faces);

  print_grid_data(pcout, n_refine_space, triangulation);

  comp_navier_stokes_operator->setup(boundary_descriptor_density,
                                     boundary_descriptor_velocity,
                                     boundary_descriptor_pressure,
                                     boundary_descriptor_energy,
                                     field_functions,
                                     analytical_solution);

  time_integrator->setup();
  time_integrator->timeloop();
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
        CompressibleNavierStokesProblem<DIMENSION, FE_DEGREE, QPOINTS_CONV, QPOINTS_VIS>
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
