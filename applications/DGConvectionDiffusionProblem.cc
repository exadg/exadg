/*
 * DGConvectionDiffusionProblem.cc
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

#include "../include/DGConvDiffOperation.h"

#include "../include/TimeIntExplRKConvDiff.h"
#include "../include/TimeIntBDFConvDiff.h"

#include "../include/BoundaryDescriptorConvDiff.h"
#include "../include/FieldFunctionsConvDiff.h"
#include "../include/InputParametersConvDiff.h"

#include "../include/PrintFunctions.h"

#include "../include/PostProcessorConvDiff.h"
#include "../include/AnalyticalSolutionConvDiff.h"

using namespace dealii;
using namespace ConvDiff;


// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

//#include "ConvectionDiffusionTestCases/PropagatingSineWave.h"
//#include "ConvectionDiffusionTestCases/RotatingHill.h"
#include "ConvectionDiffusionTestCases/DeformingHill.h"
//#include "ConvectionDiffusionTestCases/DiffusiveProblemHomogeneousDBC.h"
//#include "ConvectionDiffusionTestCases/DiffusiveProblemHomogeneousNBC.h"
//#include "ConvectionDiffusionTestCases/DiffusiveProblemHomogeneousNBC2.h"
//#include "ConvectionDiffusionTestCases/ConstantRHS.h"
//#include "ConvectionDiffusionTestCases/BoundaryLayerProblem.h"



template<int dim, int fe_degree>
class ConvDiffProblem
{
public:
  typedef double value_type;
  ConvDiffProblem(const unsigned int n_refine_space,
                  const unsigned int n_refine_time);

  void solve_problem();

private:
  void print_grid_data();

  void setup_postprocessor();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  InputParametersConvDiff param;

  const unsigned int n_refine_space;
  const unsigned int n_refine_time;

  std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions;
  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;

  std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> > analytical_solution;

  std_cxx11::shared_ptr<DGConvDiffOperation<dim,fe_degree, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<ConvDiff::PostProcessor<dim, fe_degree> > postprocessor;

  std_cxx11::shared_ptr<TimeIntExplRKConvDiff<dim, fe_degree, value_type> > time_integrator_explRK;
  std_cxx11::shared_ptr<TimeIntBDFConvDiff<dim,fe_degree,value_type> > time_integrator_BDF;
};

template<int dim, int fe_degree>
ConvDiffProblem<dim, fe_degree>::
ConvDiffProblem(const unsigned int n_refine_space_in,
                const unsigned int n_refine_time_in)
  :
  pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,
                dealii::Triangulation<dim>::none,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(n_refine_space_in),
  n_refine_time(n_refine_time_in)
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                     unsteady convection-diffusion equation                      " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;

  param.set_input_parameters();
  param.check_input_parameters();

  if(param.print_input_parameters)
    param.print(pcout);

  field_functions.reset(new FieldFunctionsConvDiff<dim>());
  // this function has to be defined in the header file that implements
  // all problem specific things like parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolutionConvDiff<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor.reset(new BoundaryDescriptorConvDiff<dim>());

  // initialize convection diffusion operation
  conv_diff_operation.reset(new DGConvDiffOperation<dim, fe_degree, value_type>(triangulation,param));

  // initialize postprocessor
  postprocessor.reset(new ConvDiff::PostProcessor<dim, fe_degree>());

  // initialize time integrator
  if(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
  {
    time_integrator_explRK.reset(new TimeIntExplRKConvDiff<dim, fe_degree, value_type>(
        conv_diff_operation,
        postprocessor,
        param,
        field_functions->velocity,
        n_refine_time));
  }
  else if(param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
  {
    time_integrator_BDF.reset(new TimeIntBDFConvDiff<dim, fe_degree, value_type>(
        conv_diff_operation,
        postprocessor,
        param,
        field_functions->velocity,
        n_refine_time));
  }
  else
  {
    AssertThrow(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK ||
                param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF,
                ExcMessage("Specified time integration scheme is not implemented!"));
  }
}

template<int dim, int fe_degree>
void ConvDiffProblem<dim, fe_degree>::
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

template<int dim, int fe_degree>
void ConvDiffProblem<dim, fe_degree>::
setup_postprocessor()
{
  ConvDiff::PostProcessorData pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;

  postprocessor->setup(pp_data,
                       conv_diff_operation->get_dof_handler(),
                       conv_diff_operation->get_mapping(),
                       conv_diff_operation->get_data(),
                       analytical_solution);
}

template<int dim, int fe_degree>
void ConvDiffProblem<dim, fe_degree>::
solve_problem()
{
  // this function has to be defined in the header file that implements
  // all problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor);

  print_grid_data();

  conv_diff_operation->setup(periodic_faces,
                             boundary_descriptor,
                             field_functions);

  setup_postprocessor();

  if(param.temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
  {
    time_integrator_explRK->setup();

    time_integrator_explRK->timeloop();
  }
  else if(param.temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
  {
    // call setup() of time_integrator before setup_solvers() of conv_diff_operation
    // because setup_solvers() needs quantities such as the time step size for a
    // correct initialization of preconditioners
    time_integrator_BDF->setup();

    conv_diff_operation->setup_solver();

    time_integrator_BDF->timeloop();
  }
}

int main (int argc, char** argv)
{
  try
  {
    //using namespace ConvectionDiffusionProblem;
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX; ++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;
          refine_steps_time <= REFINE_STEPS_TIME_MAX; ++refine_steps_time)
      {
        ConvDiffProblem<DIMENSION, FE_DEGREE> conv_diff_problem(refine_steps_space,
                                                                refine_steps_time);
        conv_diff_problem.solve_problem();
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
