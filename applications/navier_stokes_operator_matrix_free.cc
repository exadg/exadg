/*
 * navier_stokes_operator_matrix_free.cc
 *
 *  Created on: May 5, 2017
 *      Author: fehn
 */

// deal.ii

// triangulation
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// timer
#include <deal.II/base/timer.h>

// ExaDG
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"

// Navier-Stokes operator
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"

using namespace dealii;


// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/couette.h"
//#include "incompressible_navier_stokes_test_cases/poiseuille.h"
#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/stokes_guermond.h"
//#include "incompressible_navier_stokes_test_cases/stokes_shahbazi.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
//#include "incompressible_navier_stokes_test_cases/vortex.h"
//#include "incompressible_navier_stokes_test_cases/taylor_vortex.h"
//#include "incompressible_navier_stokes_test_cases/beltrami.h"
//#include "incompressible_navier_stokes_test_cases/flow_past_cylinder.h"
//#include "incompressible_navier_stokes_test_cases/turbulent_channel.h"

/**************************************************************************************/
/*                                                                                    */
/*                          FURTHER INPUT PARAMETERS                                  */
/*                                                                                    */
/**************************************************************************************/

// set the polynomial degree of the shape functions k = 1,...,10
unsigned int const FE_DEGREE_U_MIN = FE_DEGREE_VELOCITY;
unsigned int const FE_DEGREE_U_MAX = 10;

// Decide whether to apply the nonlinear Navier-Stokes operator or the
// linearized operator.
enum class OperatorType{
  Nonlinear,
  Linearized
};

OperatorType OPERATOR_TYPE = OperatorType::Linearized; //Nonlinear; //Linearized;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS = 100;

// Type of wall time calculation used to measure efficiency
enum class WallTimeCalculation{
  Average,
  Minimum
};

WallTimeCalculation const WALL_TIME_CALCULATION = WallTimeCalculation::Minimum; //Average; //Minimum;

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double> > wall_times;


template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class NavierStokesProblem
{
public:
  NavierStokesProblem(unsigned int const refine_steps_space);
  void setup();
  void apply_operator();

private:
  void print_header();
  void print_mpi_info();
  void print_grid_data();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

  unsigned int const n_refine_space;

  std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  InputParametersNavierStokes<dim> param;

  std::shared_ptr<DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation;

  // number of matrix-vector products
  unsigned int const n_repetitions;

  // wall time calculation
  WallTimeCalculation const wall_time_calculation;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
NavierStokesProblem(unsigned int const refine_steps_space)
  :
  pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,
                dealii::Triangulation<dim>::none,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space),
  n_repetitions(N_REPETITIONS),
  wall_time_calculation(WALL_TIME_CALCULATION)
{
  print_header();
  print_mpi_info();

  param.set_input_parameters();
  param.check_input_parameters();

  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new FieldFunctionsNavierStokes<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolutionNavierStokes<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

  // initialize navier_stokes_operation
  navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
      (triangulation,param));
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
print_header()
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin discretization                 " << std::endl
  << "                  of the incompressible Navier-Stokes equations                  " << std::endl
  << "                      based on a matrix-free implementation                      " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
print_mpi_info()
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout,"Number of processes",Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
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

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup()
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

  navier_stokes_operation->setup_velocity_conv_diff_operator();
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // initialize vectors
  parallel::distributed::BlockVector<VALUE_TYPE> dst, src;
  navier_stokes_operation->initialize_block_vector_velocity_pressure(dst);
  navier_stokes_operation->initialize_block_vector_velocity_pressure(src);
  src = 1.0;

  // Timer and wall times
  Timer timer;

  double wall_time = 0.0;

  if(wall_time_calculation == WallTimeCalculation::Minimum)
    wall_time = std::numeric_limits<double>::max();

  // set linearized solution -> required to evaluate the linearized operator
  navier_stokes_operation->set_solution_linearization(src);
  // set sum_alphai_ui -> required to evaluate the nonlinear residual
  // in case an unsteady problem is considered
  navier_stokes_operation->set_sum_alphai_ui(&src.block(0));

  // apply matrix-vector product several times
  for(unsigned int i=0; i<n_repetitions; ++i)
  {
    timer.restart();

    if(OPERATOR_TYPE == OperatorType::Nonlinear)
      navier_stokes_operation->evaluate_nonlinear_residual(dst,src);
    else if(OPERATOR_TYPE == OperatorType::Linearized)
      navier_stokes_operation->vmult(dst,src);

    double const current_wall_time = timer.wall_time();

    if(wall_time_calculation == WallTimeCalculation::Average)
      wall_time += current_wall_time;
    else if(wall_time_calculation == WallTimeCalculation::Minimum)
      wall_time = std::min(wall_time,current_wall_time);
  }
  // reset sum_alphai_ui
  navier_stokes_operation->set_sum_alphai_ui(nullptr);

  // compute wall times
  if(wall_time_calculation == WallTimeCalculation::Average)
    wall_time /= (double)n_repetitions;

  unsigned int dofs =   navier_stokes_operation->get_dof_handler_u().n_dofs()
                      + navier_stokes_operation->get_dof_handler_p().n_dofs();

  wall_time /= (double) dofs;

  pcout << std::endl << std::scientific << std::setprecision(4)
        << "Wall time / dofs [s]: " << wall_time << std::endl;

  wall_times.push_back(std::pair<unsigned int,double>(fe_degree_u,wall_time));

  pcout << std::endl << " ... done." << std::endl << std::endl;
}


/**************************************************************************************/
/*                                                                                    */
/*                                         MAIN                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Precompile NavierStokesProblem for all polynomial degrees in the range
 *  FE_DEGREE_U_MIN < fe_degree_i < FE_DEGREE_U_MAX so that we do not have to recompile
 *  in order to run the program for different polynomial degrees
 */
template<int dim, int fe_degree_u, int max_fe_degree_u, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class NavierStokesPrecompiled
{
public:
  static void run(unsigned int const refine_steps_space)
  {
    NavierStokesPrecompiled<dim,fe_degree_u,fe_degree_u,fe_degree_xwall,xwall_quad_rule,Number>::run(refine_steps_space);
    NavierStokesPrecompiled<dim,fe_degree_u+1,max_fe_degree_u,fe_degree_xwall,xwall_quad_rule,Number>::run(refine_steps_space);
  }
};

/*
 * specialization of templates: fe_degree_u == max_fe_degree_u
 * Note that fe_degree_p = fe_degree_u - 1.
 */
template <int dim, int fe_degree_u, int fe_degree_xwall, int xwall_quad_rule,typename Number>
class NavierStokesPrecompiled<dim, fe_degree_u, fe_degree_u, fe_degree_xwall, xwall_quad_rule, Number>
{
public:
  static void run(unsigned int const refine_steps_space)
  {
    typedef NavierStokesProblem<dim,fe_degree_u,fe_degree_u-1 /* fe_degree_p*/,
      fe_degree_xwall,xwall_quad_rule,Number> NAVIER_STOKES_PROBLEM;

    NAVIER_STOKES_PROBLEM navier_stokes_problem(refine_steps_space);
    navier_stokes_problem.setup();
    navier_stokes_problem.apply_operator();
  }
};

void print_wall_times(std::vector<std::pair<unsigned int, double> > const &wall_times,
                      unsigned int const                                  refine_steps_space)
{
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Wall times for refine level l = " << refine_steps_space << ":"
              << std::endl << std::endl
              << "  k    " << "wall time / dofs [s]" << std::endl;

    typedef typename std::vector<std::pair<unsigned int, double> >::const_iterator ITERATOR;
    for(ITERATOR it=wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << "  " << std::setw(5) << std::left << it->first
                << std::setw(2) << std::left << std::scientific << std::setprecision(4) << it->second
                << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
  }
}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    //mesh refinements
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX;++refine_steps_space)
    {
      // increasing polynomial degrees
      typedef NavierStokesPrecompiled<DIMENSION,FE_DEGREE_U_MIN,FE_DEGREE_U_MAX,
          FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,VALUE_TYPE> NAVIER_STOKES;

      NAVIER_STOKES::run(refine_steps_space);

      print_wall_times(wall_times, refine_steps_space);
      wall_times.clear();
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



