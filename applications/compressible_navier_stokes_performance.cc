/*
 * compressible_navier_stokes_performance.cc
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

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

#include "compressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"

/**************************************************************************************/
/*                                                                                    */
/*                          FURTHER INPUT PARAMETERS                                  */
/*                                                                                    */
/**************************************************************************************/

// set the polynomial degree k of the shape functions
unsigned int const FE_DEGREE_MIN = 2;
unsigned int const FE_DEGREE_MAX = 2;

// refinement level: l = REFINE_LEVELS[degree-1]
std::vector<int> REFINE_LEVELS = {
  7, /* k=1 */
  6,
  6, /* k=3 */
  5,
  5,
  5,
  5, /* k=7 */
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4 /* k=15 */
};

// NOTE: the quadrature rule specified in the parameter file is irrelevant for these
//       performance measurements. The quadrature rule has to be selected manually
//       at the bottom of this file!

// Select the operator to be applied
enum class OperatorType
{
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms,
  InverseMassMatrix,
  InverseMassMatrixDstDst,
  VectorUpdate,
  EvaluateOperatorExplicit
};

OperatorType OPERATOR_TYPE =
  OperatorType::ViscousAndConvectiveTerms; // EvaluateOperatorExplicit; //ViscousAndConvectiveTerms;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 1;   // take the minimum of outer repetitions

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double>> wall_times;

using namespace dealii;
using namespace CompNS;

namespace CompNS
{
template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number = double>
class Problem
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DGOperator<dim, degree, n_q_points_conv, n_q_points_vis, Number> DG_OPERATOR;

  typedef TimeIntExplRK<dim, Number> TIME_INT;

  typedef PostProcessor<dim, degree, n_q_points_conv, n_q_points_vis, Number> POSTPROCESSOR;

  Problem(unsigned int const refine_steps_space, unsigned int const refine_steps_time = 0);

  void
  setup();

  void
  apply_operator();

private:
  void
  print_header();

  ConditionalOStream pcout;

  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  const unsigned int n_refine_space;
  const unsigned int n_refine_time;

  std::shared_ptr<FieldFunctions<dim>>           field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;
  std::shared_ptr<AnalyticalSolution<dim>>       analytical_solution;

  InputParameters<dim> param;

  std::shared_ptr<DG_OPERATOR> comp_navier_stokes_operator;

  std::shared_ptr<POSTPROCESSOR> postprocessor;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;
};

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
Problem<dim, degree, n_q_points_conv, n_q_points_vis, Number>::Problem(
  unsigned int const n_refine_space_in,
  unsigned int const n_refine_time_in)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_refine_space(n_refine_space_in),
    n_refine_time(n_refine_time_in),
    n_repetitions_inner(N_REPETITIONS_INNER),
    n_repetitions_outer(N_REPETITIONS_OUTER)
{
  param.set_input_parameters();
  param.check_input_parameters();

  print_header();
  print_MPI_info(pcout);
  if(param.print_input_parameters == true)
    param.print(pcout);

  // triangulation
  if(param.triangulation_type == TriangulationType::Distributed)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor_density.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_velocity.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_energy.reset(new BoundaryDescriptorEnergy<dim>());

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor =
    construct_postprocessor<dim, degree, n_q_points_conv, n_q_points_vis, Number>(param);

  // initialize compressible Navier-Stokes operator
  comp_navier_stokes_operator.reset(new DG_OPERATOR(*triangulation, param, postprocessor));
}

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
void
Problem<dim, degree, n_q_points_conv, n_q_points_vis, Number>::print_header()
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

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
void
Problem<dim, degree, n_q_points_conv, n_q_points_vis, Number>::setup()
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

  print_grid_data(pcout, n_refine_space, *triangulation);

  comp_navier_stokes_operator->setup(boundary_descriptor_density,
                                     boundary_descriptor_velocity,
                                     boundary_descriptor_pressure,
                                     boundary_descriptor_energy,
                                     field_functions,
                                     analytical_solution);
}

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
void
Problem<dim, degree, n_q_points_conv, n_q_points_vis, Number>::apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // Vectors
  VectorType dst, src;

  // initialize vectors
  comp_navier_stokes_operator->initialize_dof_vector(src);
  comp_navier_stokes_operator->initialize_dof_vector(dst);
  src = 1.0;
  dst = 1.0;

  // Timer and wall times
  Timer  timer;
  double wall_time = std::numeric_limits<double>::max();

  for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
  {
    double current_wall_time = 0.0;

    // apply matrix-vector product several times
    for(unsigned int i = 0; i < n_repetitions_inner; ++i)
    {
      timer.restart();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("compressible_deg_" + std::to_string(degree)).c_str());
#endif

      if(OPERATOR_TYPE == OperatorType::ConvectiveTerm)
        comp_navier_stokes_operator->evaluate_convective(dst, src, 0.0);
      else if(OPERATOR_TYPE == OperatorType::ViscousTerm)
        comp_navier_stokes_operator->evaluate_viscous(dst, src, 0.0);
      else if(OPERATOR_TYPE == OperatorType::ViscousAndConvectiveTerms)
        comp_navier_stokes_operator->evaluate_convective_and_viscous(dst, src, 0.0);
      else if(OPERATOR_TYPE == OperatorType::InverseMassMatrix)
        comp_navier_stokes_operator->apply_inverse_mass(dst, src);
      else if(OPERATOR_TYPE == OperatorType::InverseMassMatrixDstDst)
        comp_navier_stokes_operator->apply_inverse_mass(dst, dst);
      else if(OPERATOR_TYPE == OperatorType::VectorUpdate)
        dst.sadd(2.0, 1.0, src);
      else if(OPERATOR_TYPE == OperatorType::EvaluateOperatorExplicit)
        comp_navier_stokes_operator->evaluate(dst, src, 0.0);
      else
        AssertThrow(false, ExcMessage("Specified operator type not implemented"));

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("compressible_deg_" + std::to_string(degree)).c_str());
#endif

      Utilities::MPI::MinMaxAvg wall_time =
        Utilities::MPI::min_max_avg(timer.wall_time(), MPI_COMM_WORLD);

      current_wall_time += wall_time.avg;
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time, current_wall_time);
  }

  if(wall_time * n_repetitions_inner * n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout
      << std::endl
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducable results."
      << std::endl;
  }

  unsigned int dofs = comp_navier_stokes_operator->get_dof_handler().n_dofs();

  double wall_time_per_dofs = wall_time / (double)dofs;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4) << "t_wall/DoF [s]:  " << wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/sec:        " << 1. / wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/(sec*core): " << 1. / wall_time_per_dofs / (double)N_mpi_processes << std::endl;
  // clang-format on

  wall_times.push_back(std::pair<unsigned int, double>(degree, wall_time_per_dofs));

  pcout << std::endl << " ... done." << std::endl << std::endl;
}

} // namespace CompNS

void
print_wall_times(std::vector<std::pair<unsigned int, double>> const & wall_times)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::string str_operator_type[] = {"ConvectiveTerm",
                                       "ViscousTerm",
                                       "ViscousAndConvectiveTerms",
                                       "InverseMassMatrix",
                                       "InverseMassMatrixDstDst",
                                       "VectorUpdate",
                                       "EvaluateOperatorExplicit"};

    // clang-format off
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Operator type: " << str_operator_type[(int)OPERATOR_TYPE] << std::endl
              << std::endl
              << "Wall times and throughput:" << std::endl
              << std::endl
              << "  k    " << "t_wall/DoF [s] " << "DoFs/sec   " << "DoFs/(sec*core) " << std::endl;

    typedef typename std::vector<std::pair<unsigned int, double>>::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << std::scientific << std::setprecision(4)
                << "  " << std::setw(5) << std::left << it->first
                << std::setw(2) << std::left << it->second
                << "     " << std::setw(2) << std::left << 1. / it->second
                << " " << std::setw(2) << std::left << 1. / it->second / (double)N_mpi_processes
                << std::endl;
    }
    // clang-format on

    std::cout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
  }
}

/**************************************************************************************/
/*                                                                                    */
/*                                         MAIN                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Precompile NavierStokesProblem for all polynomial degrees in the range
 *  FE_DEGREE_MIN < degree_i < FE_DEGREE_MAX so that we do not have to recompile
 *  in order to run the program for different polynomial degrees
 */
template<int dim, int degree, int max_degree>
class NavierStokesPrecompiled
{
public:
  static void
  run()
  {
    NavierStokesPrecompiled<dim, degree, degree>::run();
    NavierStokesPrecompiled<dim, degree + 1, max_degree>::run();
  }
};

/*
 * specialization of templates: degree == max_degree
 */
template<int dim, int degree>
class NavierStokesPrecompiled<dim, degree, degree>
{
public:
  /*
   *  Select the quadrature formula manually for convective term
   */

  // standard quadrature
  //  static const unsigned int n_q_points_conv = degree+1;

  // 3/2 dealiasing rule
  //  static const unsigned int n_q_points_conv = degree+(degree+2)/2;

  // 2k dealiasing rule
  static const unsigned int n_q_points_conv = 2 * degree + 1;

  /*
   *  Select the quadrature formula manually for viscous term
   */

  // standard quadrature
  //  static const unsigned int n_q_points_vis = degree+1;

  // same as convective term
  static const unsigned int n_q_points_vis = n_q_points_conv;


  // setup problem and apply operator
  static void
  run()
  {
    typedef Problem<dim, degree, n_q_points_conv, n_q_points_vis> PROBLEM;

    PROBLEM problem(REFINE_LEVELS[degree - 1]);
    problem.setup();
    problem.apply_operator();
  }
};

int
main(int argc, char ** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

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

    // measure throughput for increasing polynomial degree
    typedef NavierStokesPrecompiled<DIMENSION, FE_DEGREE_MIN, FE_DEGREE_MAX> NAVIER_STOKES;
    NAVIER_STOKES::run();

    print_wall_times(wall_times);
    wall_times.clear();
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
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
