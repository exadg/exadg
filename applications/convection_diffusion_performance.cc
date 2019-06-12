/*
 * convection_diffusion.cc
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// spatial discretization
#include "../include/convection_diffusion/spatial_discretization/dg_operator.h"
#include "../include/convection_diffusion/spatial_discretization/interface.h"

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor_base.h"

// user interface, etc.
#include "../include/functionalities/print_throughput.h"
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"


// specify the test case that has to be solved

// template
#include "convection_diffusion_test_cases/periodic_box.h"

using namespace dealii;
using namespace ConvDiff;

// refinement level: l = REFINE_LEVELS[fe_degree-1]
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

// Select the operator to be applied

enum class Operator
{
  MassOperator,
  ConvectiveOperator,
  DiffusiveOperator,
  MassConvectionDiffusionOperator
};

Operator OPERATOR = Operator::MassConvectionDiffusionOperator;

std::string
enum_to_string(Operator const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operator::MassOperator:                    string_type = "MassOperator";                    break;
    case Operator::ConvectiveOperator:              string_type = "ConvectiveOperator";              break;
    case Operator::DiffusiveOperator:               string_type = "DiffusiveOperator";               break;
    case Operator::MassConvectionDiffusionOperator: string_type = "MassConvectionDiffusionOperator"; break;

    default:AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 1;   // take the minimum of outer repetitions

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double>> wall_times;

template<typename Number>
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

template<int dim, typename Number = double>
class Problem : public ProblemBase<Number>
{
public:
  Problem();

  void
  setup(InputParameters const & param);

  void
  apply_operator();

private:
  void
  print_header();

  ConditionalOStream pcout;

  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<DGOperator<dim, Number>> conv_diff_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

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
  << "                          convection-diffusion equation                          " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Problem<dim, Number>::setup(InputParameters const & param_in)
{
  print_header();
  print_dealii_info<Number>(pcout);
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

  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  set_boundary_conditions(boundary_descriptor);

  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, Number>(param);

  // initialize convection diffusion operation
  conv_diff_operator.reset(new DGOperator<dim, Number>(*triangulation, param, postprocessor));
  conv_diff_operator->setup(periodic_faces, boundary_descriptor, field_functions);
  conv_diff_operator->setup_solver(1.0 /* use a default value of 1.0 */);
}

template<int dim, typename Number>
void
Problem<dim, Number>::apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  LinearAlgebra::distributed::Vector<Number> dst, src, velocity;

  conv_diff_operator->initialize_dof_vector(src);
  conv_diff_operator->initialize_dof_vector(dst);

  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    if(param.type_velocity_field == TypeVelocityField::Numerical)
      conv_diff_operator->initialize_dof_vector_velocity(velocity);

    velocity = 1.0;
  }

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

      if(OPERATOR == Operator::MassOperator)
        conv_diff_operator->apply_mass_matrix(dst, src);
      else if(OPERATOR == Operator::ConvectiveOperator)
        conv_diff_operator->apply_convective_term(dst, src, 1.0 /* time */, &velocity);
      else if(OPERATOR == Operator::DiffusiveOperator)
        conv_diff_operator->apply_diffusive_term(dst, src);
      else if(OPERATOR == Operator::MassConvectionDiffusionOperator)
        conv_diff_operator->apply(
          dst, src, 1.0 /* time */, 1.0 /* scaling_factor_mass_matrix */, &velocity);

      current_wall_time += timer.wall_time();
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time, current_wall_time);
  }

  if(wall_time * n_repetitions_inner * n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout
      << std::endl
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducible results."
      << std::endl;
  }

  types::global_dof_index dofs              = conv_diff_operator->get_number_of_dofs();
  double                  dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  wall_times.push_back(std::pair<unsigned int, double>(param.degree, dofs_per_walltime));

  pcout << std::endl << " ... done." << std::endl << std::endl;
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    // set parameters
    ConvDiff::InputParameters param;
    set_input_parameters(param);

    // k-refinement
    for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
    {
      // reset degree
      param.degree = degree;

      // reset h-refinements
      param.h_refinements = REFINE_LEVELS[degree - 1];

      // setup problem and run simulation
      typedef double                       Number;
      std::shared_ptr<ProblemBase<Number>> problem;

      if(param.dim == 2)
        problem.reset(new Problem<2, Number>());
      else if(param.dim == 3)
        problem.reset(new Problem<3, Number>());
      else
        AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

      problem->setup(param);

      problem->apply_operator();
    }

    print_throughput(wall_times, enum_to_string(OPERATOR));
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
  return 0;
}
