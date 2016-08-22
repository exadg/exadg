
// Navier-Stokes splitting program
// authors: Niklas Fehn, Benjamin Krank, Martin Kronbichler, LNM
// years: 2015-2016

#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>
#include <sstream>

#include "../include/DGNavierStokesDualSplitting.h"
#include "../include/DGNavierStokesDualSplittingXWall.h"
#include "../include/DGNavierStokesCoupled.h"

#include "../include/InputParametersNavierStokes.h"
#include "TimeIntBDFDualSplitting.h"
#include "TimeIntBDFDualSplittingXWall.h"
#include "TimeIntBDFDualSplittingXWallSpalartAllmaras.h"
#include "TimeIntBDFCoupled.h"
#include "../include/PostProcessor.h"
#include "PostProcessorXWall.h"
#include "PrintInputParameters.h"

#include "DriverSteadyProblems.h"

#include "../include/BoundaryDescriptorNavierStokes.h"
#include "../include/FieldFunctionsNavierStokes.h"

using namespace dealii;

// specify the flow problem that has to be solved

//#include "NavierStokesTestCases/Cuette.h"
//#include "NavierStokesTestCases/Poiseuille.h"
//#include "NavierStokesTestCases/Cavity.h"
//#include "NavierStokesTestCases/StokesGuermond.h"
//#include "NavierStokesTestCases/StokesShahbazi.h"
//#include "NavierStokesTestCases/Kovasznay.h"
//#include "NavierStokesTestCases/Vortex.h"
//#include "NavierStokesTestCases/TaylorVortex.h"
//#include "NavierStokesTestCases/Beltrami.h"
//#include "NavierStokesTestCases/FlowPastCylinder.h"


template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class NavierStokesProblem
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
  void solve_problem(bool do_restart);

private:
  void print_header();
  void print_grid_data();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  const unsigned int n_refine_space;

  std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;

  InputParametersNavierStokes param;

  std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > navier_stokes_operation;

  std_cxx11::shared_ptr<PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > postprocessor;

  std_cxx11::shared_ptr<TimeIntBDF<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> > time_integrator;

  std_cxx11::shared_ptr<DriverSteadyProblems<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> > driver_steady;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
NavierStokesProblem(unsigned int const refine_steps_space,
                    unsigned int const refine_steps_time)
  :
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
{
  param.set_input_parameters();
  param.check_input_parameters();

  print_header();
  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new FieldFunctionsNavierStokes<dim>());
  // this function has to be defined in the header file that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

  if(param.spatial_discretization == SpatialDiscretization::DGXWall)
  {
    if(param.problem_type == ProblemType::Unsteady &&
            param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      // initialize navier_stokes_operation
      navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
          (triangulation,param));
      // initialize postprocessor after initializing navier_stokes_operation
      postprocessor.reset(new PostProcessorXWall<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(navier_stokes_operation,param));
      // initialize time integrator that depends on both navier_stokes_operation and postprocessor
      time_integrator.reset(new TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
          navier_stokes_operation,postprocessor,param,refine_steps_time));
    }
    else
    {
      AssertThrow(false,ExcMessage("XWall only implemented for the unsteady DualSplitting case up to now"));
    }
  }
  else if(param.spatial_discretization == SpatialDiscretization::DG)
  {
    if(param.problem_type == ProblemType::Steady)
    {
      // initialize navier_stokes_operation
      navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
          (triangulation,param));
      // initialize postprocessor after initializing navier_stokes_operation
      postprocessor.reset(new PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(navier_stokes_operation,param));
      // initialize driver for steady state problem that depends on both navier_stokes_operation and postprocessor
      driver_steady.reset(new DriverSteadyProblems<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
          (navier_stokes_operation,postprocessor,param));

    }
    else if(param.problem_type == ProblemType::Unsteady &&
            param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      // initialize navier_stokes_operation
      navier_stokes_operation.reset(new DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
          (triangulation,param));
      // initialize postprocessor after initializing navier_stokes_operation
      postprocessor.reset(new PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(navier_stokes_operation,param));
      // initialize time integrator that depends on both navier_stokes_operation and postprocessor
      time_integrator.reset(new TimeIntBDFDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
          navier_stokes_operation,postprocessor,param,refine_steps_time));
    }
    else if(param.problem_type == ProblemType::Unsteady &&
            param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      // initialize navier_stokes_operation
      navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
          (triangulation,param));
      // initialize postprocessor after initializing navier_stokes_operation
      postprocessor.reset(new PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(navier_stokes_operation,param));
      // initialize time integrator that depends on both navier_stokes_operation and postprocessor
      time_integrator.reset(new TimeIntBDFCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
          navier_stokes_operation,postprocessor,param,refine_steps_time));
    }
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
print_header()
{
  if(param.problem_type == ProblemType::Steady)
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
  else if(param.problem_type == ProblemType::Unsteady)
  {
    if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      pcout << std::endl << std::endl << std::endl
      << "_________________________________________________________________________________" << std::endl
      << "                                                                                 " << std::endl
      << "                High-order discontinuous Galerkin solver for the                 " << std::endl
      << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
      << "                based on a semi-explicit dual-splitting approach                 " << std::endl
      << "_________________________________________________________________________________" << std::endl
      << std::endl;
    }
    else if(param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
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
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
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

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_problem(bool do_restart)
{
  // this function has to be defined in the header file that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure);
  print_grid_data();

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  if(param.problem_type == ProblemType::Unsteady)
  {
    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!!!)
    time_integrator->setup(do_restart);

    navier_stokes_operation->setup_solvers();

    postprocessor->setup();

    time_integrator->timeloop();
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->setup();

    navier_stokes_operation->setup_solvers();

    postprocessor->setup();

    driver_steady->solve_steady_problem();
  }
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
