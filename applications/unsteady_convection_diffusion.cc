/*
 * unsteady_convection_diffusion.cc
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */


#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor.h"

// spatial discretization
#include "convection_diffusion/spatial_discretization/dg_convection_diffusion_operation.h"

// time integration
#include "convection_diffusion/time_integration/time_int_bdf.h"
#include "convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"

// user interface, etc.
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"


// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// convection problems

//#include "convection_diffusion_test_cases/propagating_sine_wave.h"
#include "convection_diffusion_test_cases/rotating_hill.h"
//#include "convection_diffusion_test_cases/deforming_hill.h"

// diffusion problems

//#include "convection_diffusion_test_cases/diffusive_problem.h"

// convection-diffusion problems

//#include "convection_diffusion_test_cases/constant_rhs.h"
//#include "convection_diffusion_test_cases/boundary_layer_problem.h"
//#include "convection_diffusion_test_cases/const_rhs_const_and_circular_wind.h"

using namespace dealii;
using namespace ConvDiff;

template<int dim, int fe_degree, typename Number = double>
class ConvDiffProblem
{
public:
  typedef double value_type;
  ConvDiffProblem(const unsigned int n_refine_space, const unsigned int n_refine_time);

  void
  solve_problem();

private:
  void
  print_header();

  void
  print_grid_data();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  ConvDiff::InputParameters param;

  const unsigned int n_refine_space;
  const unsigned int n_refine_time;

  std::shared_ptr<ConvDiff::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<ConvDiff::AnalyticalSolution<dim>> analytical_solution;

  std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation;
  std::shared_ptr<ConvDiff::PostProcessor<dim, fe_degree>>           postprocessor;

  std::shared_ptr<ConvDiff::TimeIntExplRK<dim, fe_degree, value_type>> time_integrator_explRK;
  std::shared_ptr<ConvDiff::TimeIntBDF<dim, fe_degree, value_type>>    time_integrator_BDF;
};

template<int dim, int fe_degree, typename Number>
ConvDiffProblem<dim, fe_degree, Number>::ConvDiffProblem(const unsigned int n_refine_space_in,
                                                         const unsigned int n_refine_time_in)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space(n_refine_space_in),
    n_refine_time(n_refine_time_in)
{
  param.set_input_parameters();
  param.check_input_parameters();
  AssertThrow(param.problem_type == ProblemType::Unsteady,
              ExcMessage("ProblemType must be unsteady!"));

  print_header();
  print_MPI_info(pcout);
  if(param.print_input_parameters)
    param.print(pcout);

  field_functions.reset(new ConvDiff::FieldFunctions<dim>());
  // this function has to be defined in the header file that implements
  // all problem specific things like parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  analytical_solution.reset(new ConvDiff::AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<dim>());

  // initialize postprocessor
  postprocessor.reset(new ConvDiff::PostProcessor<dim, fe_degree>());

  // initialize convection diffusion operation
  conv_diff_operation.reset(
    new ConvDiff::DGOperation<dim, fe_degree, value_type>(triangulation, param, postprocessor));

  // initialize time integrator
  if(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
  {
    time_integrator_explRK.reset(new ConvDiff::TimeIntExplRK<dim, fe_degree, value_type>(
      conv_diff_operation, param, field_functions->velocity, n_refine_time));
  }
  else if(param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
  {
    time_integrator_BDF.reset(new ConvDiff::TimeIntBDF<dim, fe_degree, value_type>(
      conv_diff_operation, param, field_functions->velocity, n_refine_time));
  }
  else
  {
    AssertThrow(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK ||
                  param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF,
                ExcMessage("Specified time integration scheme is not implemented!"));
  }
}

template<int dim, int fe_degree, typename Number>
void
ConvDiffProblem<dim, fe_degree, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                     unsteady convection-diffusion equation                      " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, int fe_degree, typename Number>
void
ConvDiffProblem<dim, fe_degree, Number>::print_grid_data()
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
ConvDiffProblem<dim, fe_degree, Number>::solve_problem()
{
  // this function has to be defined in the header file that implements
  // all problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation, n_refine_space, boundary_descriptor);

  print_grid_data();

  conv_diff_operation->setup(periodic_faces,
                             boundary_descriptor,
                             field_functions,
                             analytical_solution);

  if(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
  {
    time_integrator_explRK->setup();

    time_integrator_explRK->timeloop();
  }
  else if(param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
  {
    // call setup() of time_integrator before setup_solvers() of conv_diff_operation
    // because setup_solver() needs quantities such as the time step size for a
    // correct initialization of preconditioners
    time_integrator_BDF->setup();

    conv_diff_operation->setup_solver(
      time_integrator_BDF->get_scaling_factor_time_derivative_term());

    time_integrator_BDF->timeloop();
  }
}

int
main(int argc, char ** argv)
{
  try
  {
    // using namespace ConvectionDiffusionProblem;
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
        ConvDiffProblem<DIMENSION, FE_DEGREE> conv_diff_problem(refine_steps_space,
                                                                refine_steps_time);
        conv_diff_problem.solve_problem();
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
