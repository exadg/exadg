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
#include "../include/compressible_navier_stokes/postprocessor/postprocessor_base.h"

// spatial discretization
#include "../include/compressible_navier_stokes/spatial_discretization/dg_operator.h"

// temporal discretization
#include "../include/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.h"

// Parameters, BCs, etc.
#include "../include/compressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/compressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/compressible_navier_stokes/user_interface/field_functions.h"
#include "../include/compressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// specify the test case that has to be solved

#include "compressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"

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
//       in the main function.

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

OperatorType OPERATOR_TYPE = OperatorType::ConvectiveTerm; // InverseMassMatrixDstDst;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average wall time of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 1;   // take the minimum wall time of outer repetitions

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double>> wall_times;

using namespace dealii;
using namespace CompNS;

class ProblemBase
{
public:
  virtual ~ProblemBase()
  {
  }

  virtual void
  setup(InputParameters const & param) = 0;

  virtual void
  apply_operator() = 0;
};

namespace CompNS
{
template<int dim, typename Number = double>
class Problem : public ProblemBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DGOperator<dim, Number> DG_OPERATOR;

  typedef PostProcessorBase<dim, Number> POSTPROCESSOR;

  Problem();

  void
  setup(InputParameters const & param_in);

  void
  apply_operator();

private:
  void
  print_header();

  ConditionalOStream pcout;

  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<FieldFunctions<dim>>           field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;

  InputParameters param;

  std::shared_ptr<DG_OPERATOR> comp_navier_stokes_operator;

  std::shared_ptr<POSTPROCESSOR> postprocessor;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem()
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_repetitions_inner(N_REPETITIONS_INNER),
    n_repetitions_outer(N_REPETITIONS_OUTER)
{
}

template<int dim, typename Number>
void
Problem<dim, Number>::print_header()
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

template<int dim, typename Number>
void
Problem<dim, Number>::setup(InputParameters const & param_in)
{
  print_header();
  print_dealii_info(pcout);
  print_MPI_info(pcout);

  param = param_in;
  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

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

  create_grid_and_set_boundary_ids(triangulation, param.h_refinements, periodic_faces);
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor_density.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_velocity.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_energy.reset(new BoundaryDescriptorEnergy<dim>());

  CompNS::set_boundary_conditions(boundary_descriptor_density,
                                  boundary_descriptor_velocity,
                                  boundary_descriptor_pressure,
                                  boundary_descriptor_energy);

  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, Number>(param);

  // initialize compressible Navier-Stokes operator
  comp_navier_stokes_operator.reset(new DG_OPERATOR(*triangulation, param, postprocessor));

  comp_navier_stokes_operator->setup(boundary_descriptor_density,
                                     boundary_descriptor_velocity,
                                     boundary_descriptor_pressure,
                                     boundary_descriptor_energy,
                                     field_functions);
}

template<int dim, typename Number>
void
Problem<dim, Number>::apply_operator()
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

  types::global_dof_index const dofs = comp_navier_stokes_operator->get_number_of_dofs();

  double wall_time_per_dofs = wall_time / (double)dofs;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4) << "t_wall/DoF [s]:  " << wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/sec:        " << 1. / wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/(sec*core): " << 1. / wall_time_per_dofs / (double)N_mpi_processes << std::endl;
  // clang-format on

  wall_times.push_back(std::pair<unsigned int, double>(param.degree, wall_time_per_dofs));

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

int
main(int argc, char ** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    InputParameters param;
    set_input_parameters(param);

    for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
    {
      // manipulate polynomial degree
      param.degree = degree;

      // setup problem and run simulation
      typedef double               Number;
      std::shared_ptr<ProblemBase> problem;

      if(param.dim == 2)
        problem.reset(new Problem<2, Number>());
      else if(param.dim == 3)
        problem.reset(new Problem<3, Number>());
      else
        AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

      problem->setup(param);
      problem->apply_operator();
    }

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
