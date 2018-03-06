/*
 * SteadyNavierStokes.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/revision.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/driver_steady_problems.h"

// Paramters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;

// specify the flow problem that has to be solved

#include "incompressible_navier_stokes_test_cases/stokes_curl_flow.h"
//#include "incompressible_navier_stokes_test_cases/couette.h"
//#include "incompressible_navier_stokes_test_cases/poiseuille.h"
//#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
//#include "incompressible_navier_stokes_test_cases/flow_past_cylinder.h"


template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number=double>
class NavierStokesProblem
{
public:
  NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
  void solve_problem();

private:
  void print_header();
  void setup_postprocessor();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  const unsigned int n_refine_space;

  std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  InputParametersNavierStokes<dim> param;

  std::shared_ptr<DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation;

  std::shared_ptr<PostProcessorBase<dim,Number> > postprocessor;

  std::shared_ptr<DriverSteadyProblems<dim, Number,
    DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> > > driver_steady;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
NavierStokesProblem(unsigned int const refine_steps_space,
                    unsigned int const /*refine_steps_time*/)
  :
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
{
  param.set_input_parameters();
  param.check_input_parameters();

  print_header();
  print_MPI_info(pcout);
  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new FieldFunctionsNavierStokes<dim>());

  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolutionNavierStokes<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_analytical_solution(analytical_solution);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokesU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokesP<dim>());

  AssertThrow(param.solver_type == SolverType::Steady,
      ExcMessage("This is a steady solver. Check input parameters."));

  // initialize navier_stokes_operation
  navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
      (triangulation,param));

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim,Number>(param);

  // initialize driver for steady state problem that depends on both navier_stokes_operation and postprocessor
  driver_steady.reset(new DriverSteadyProblems<dim, Number, DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
      (navier_stokes_operation,postprocessor,param));
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
print_header()
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                 steady, incompressible Navier-Stokes equations                  " << std::endl
  << "            based on coupled solution approach of Newton-Krylov type             " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_postprocessor()
{
  DofQuadIndexData dof_quad_index_data;
  dof_quad_index_data.dof_index_velocity = navier_stokes_operation->get_dof_index_velocity();
  dof_quad_index_data.dof_index_pressure = navier_stokes_operation->get_dof_index_pressure();
  dof_quad_index_data.quad_index_velocity = navier_stokes_operation->get_quad_index_velocity_linear();

  postprocessor->setup(navier_stokes_operation->get_dof_handler_u(),
                       navier_stokes_operation->get_dof_handler_p(),
                       navier_stokes_operation->get_mapping(),
                       navier_stokes_operation->get_data(),
                       dof_quad_index_data,
                       analytical_solution);
}


template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_problem()
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure,
                                          periodic_faces);

  print_grid_data(pcout,
                  n_refine_space,
                  triangulation);

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  driver_steady->setup();

  navier_stokes_operation->setup_solvers();

  setup_postprocessor();

  driver_steady->solve_steady_problem();

}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl << std::endl;
    }

    deallog.depth_console(0);

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;refine_steps_space <= REFINE_STEPS_SPACE_MAX;++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            navier_stokes_problem(refine_steps_space,refine_steps_time);

        navier_stokes_problem.solve_problem();
      }
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
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
