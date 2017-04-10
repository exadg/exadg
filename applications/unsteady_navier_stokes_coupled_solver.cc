/*
 * UnsteadyNavierStokesCoupledSolver.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"

using namespace dealii;

// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/couette.h"
//#include "incompressible_navier_stokes_test_cases/poiseuille.h"
//#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/stokes_guermond.h"
//#include "incompressible_navier_stokes_test_cases/stokes_shahbazi.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
//#include "incompressible_navier_stokes_test_cases/vortex.h"
//#include "incompressible_navier_stokes_test_cases/taylor_vortex.h"
//#include "incompressible_navier_stokes_test_cases/beltrami.h"
//#include "incompressible_navier_stokes_test_cases/flow_past_cylinder.h"
#include "incompressible_navier_stokes_test_cases/turbulent_channel.h"

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class NavierStokesProblem
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;
  NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
  void solve_problem(bool do_restart);

private:
  void print_header();
  void print_mpi_info();
  void print_grid_data();
  void setup_postprocessor();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  const unsigned int n_refine_space;

  std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  InputParametersNavierStokes<dim> param;

  std::shared_ptr<DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > navier_stokes_operation;

  std::shared_ptr<PostProcessorBase<dim> > postprocessor;

  std::shared_ptr<TimeIntBDFNavierStokes<dim, fe_degree_u, value_type,
    DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > > time_integrator;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
NavierStokesProblem(unsigned int const refine_steps_space,
                    unsigned int const refine_steps_time)
  :
  pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
{
  param.set_input_parameters();
  param.check_input_parameters();

  print_header();
  print_mpi_info();
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

  boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

  bool use_adaptive_time_stepping = false;
  if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
    use_adaptive_time_stepping = true;

  AssertThrow(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution,
              ExcMessage("UnsteadyNavierStokesCoupled is an unsteady solver. Hence, problem type has to be unsteady and temporal discretization has to be BDFCoupledSolution to solve this problem."))

  // initialize navier_stokes_operation
  navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
      (triangulation,param));

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor = construct_postprocessor<dim>(param);

  // initialize time integrator that depends on both navier_stokes_operation and postprocessor
  time_integrator.reset(new TimeIntBDFCoupled<dim, fe_degree_u, value_type,
      DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >
      (navier_stokes_operation,postprocessor,param,refine_steps_time,use_adaptive_time_stepping));
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
print_header()
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "            based on coupled solution approach of Newton-Krylov type             " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
print_mpi_info()
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout,"Number of processes",Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
print_grid_data()
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout,"Number of refinements",n_refine_space);
  print_parameter(pcout,"Number of cells",triangulation.n_global_active_cells());
  print_parameter(pcout,"Number of faces",triangulation.n_active_faces());
  print_parameter(pcout,"Number of vertices",triangulation.n_vertices());
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
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

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_problem(bool do_restart)
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure,
                                          periodic_faces);
  print_grid_data();

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator->setup(do_restart);

  navier_stokes_operation->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term());

  setup_postprocessor();

  time_integrator->timeloop();
}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    bool do_restart = false;
    if (argc > 1)
    {
      do_restart = std::atoi(argv[1]);
      if(do_restart)
      {
        AssertThrow(REFINE_STEPS_SPACE_MIN == REFINE_STEPS_SPACE_MAX, ExcMessage("Spatial refinement with restart not possible!"));

        //this does in principle work
        //although it doesn't make much sense
        if(REFINE_STEPS_TIME_MIN != REFINE_STEPS_TIME_MAX && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Warning: you are starting from a restart and refine the time steps!" << std::endl;
      }
    }

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;refine_steps_space <= REFINE_STEPS_SPACE_MAX;++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            navier_stokes_problem(refine_steps_space,refine_steps_time);

        navier_stokes_problem.solve_problem(do_restart);
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
