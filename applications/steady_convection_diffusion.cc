/*
 * SteadyConvectionDiffusion.cc
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

#include "../include/convection_diffusion/analytical_solution.h"
#include "../include/convection_diffusion/boundary_descriptor.h"
#include "../include/convection_diffusion/dg_convection_diffusion_operation.h"
#include "../include/convection_diffusion/driver_steady_problems.h"
#include "../include/convection_diffusion/field_functions.h"
#include "../include/convection_diffusion/input_parameters.h"
#include "../include/convection_diffusion/postprocessor.h"
#include "../include/functionalities/print_functions.h"


using namespace dealii;
using namespace ConvDiff;


// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

//#include "convection_diffusion_test_cases/boundary_layer_problem.h"
#include "convection_diffusion_test_cases/const_rhs_const_and_circular_wind.h"


template<int dim, int fe_degree>
class ConvDiffProblem
{
public:
  typedef double value_type;
  ConvDiffProblem(const unsigned int n_refine_space);

  void solve_problem();

private:
  void print_header();

  void print_grid_data();

  void setup_postprocessor();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  ConvDiff::InputParametersConvDiff param;

  const unsigned int n_refine_space;

  std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions;
  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;

  std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> > analytical_solution;

  std_cxx11::shared_ptr<DGConvDiffOperation<dim,fe_degree, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<ConvDiff::PostProcessor<dim, fe_degree> > postprocessor;

  std_cxx11::shared_ptr<DriverSteadyConvDiff<dim, fe_degree, value_type,
      DGConvDiffOperation<dim,fe_degree, value_type> > > driver_steady_conv_diff;
};

template<int dim, int fe_degree>
ConvDiffProblem<dim, fe_degree>::
ConvDiffProblem(const unsigned int n_refine_space_in)
  :
  pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,
                dealii::Triangulation<dim>::none,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(n_refine_space_in)
{
  print_header();

  param.set_input_parameters();
  param.check_input_parameters();

  if(param.print_input_parameters == true)
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

  // initialize driver for steady convection-diffusion problems
  driver_steady_conv_diff.reset(new DriverSteadyConvDiff<dim, fe_degree, value_type,
      DGConvDiffOperation<dim,fe_degree, value_type> >(conv_diff_operation,postprocessor,param));
}

template<int dim, int fe_degree>
void ConvDiffProblem<dim, fe_degree>::
print_header()
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                      steady convection-diffusion equation                       " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
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

  conv_diff_operation->setup_solver(/*no parameter since this is a steady problem*/);

  setup_postprocessor();

  driver_steady_conv_diff->setup();

  driver_steady_conv_diff->solve_problem();
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
      ConvDiffProblem<DIMENSION, FE_DEGREE> conv_diff_problem(refine_steps_space);
      conv_diff_problem.solve_problem();
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



