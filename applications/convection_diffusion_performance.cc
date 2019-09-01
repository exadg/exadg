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
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"

#include "functionalities/mesh_resolution_generator_hypercube.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"
#include "functionalities/print_throughput.h"

// specify the test case that has to be solved
#include "convection_diffusion_test_cases/periodic_box.h"

using namespace dealii;
using namespace ConvDiff;


RunType const RUN_TYPE = RunType::FixedProblemSize;

/*
 * Specify minimum and maximum problem size for
 *  RunType::FixedProblemSize
 *  RunType::IncreasingProblemSize
 */
types::global_dof_index N_DOFS_MIN = 1e4;
types::global_dof_index N_DOFS_MAX = 3e4;

/*
 * Enable hyper_cube meshes with number of cells per direction other than multiples of 2.
 * Use this only for simple hyper_cube problems and for
 *  RunType::FixedProblemSize
 *  RunType::IncreasingProblemSize
 */
#define ENABLE_SUBDIVIDED_HYPERCUBE

#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
// will be set automatically for RunType::FixedProblemSize and RunType::IncreasingProblemSize
unsigned int SUBDIVISIONS_MESH = 1;
#endif


// Select the operator to be applied

enum class Operatortype
{
  MassOperator,
  ConvectiveOperator,
  DiffusiveOperator,
  MassConvectionDiffusionOperator
};

Operatortype OPERATOR = Operatortype::MassConvectionDiffusionOperator;

std::string
enum_to_string(Operatortype const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operatortype::MassOperator:                    string_type = "MassOperator";                    break;
    case Operatortype::ConvectiveOperator:              string_type = "ConvectiveOperator";              break;
    case Operatortype::DiffusiveOperator:               string_type = "DiffusiveOperator";               break;
    case Operatortype::MassConvectionDiffusionOperator: string_type = "MassConvectionDiffusionOperator"; break;

    default:AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 1;   // take the minimum of outer repetitions

// global variable used to store the wall times for different polynomial degrees and problem sizes
std::vector<std::tuple<unsigned int, types::global_dof_index, double>> WALL_TIMES;

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

  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<DGOperator<dim, Number>> conv_diff_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;

  // velocity vector needed in case of numerical velocity field
  LinearAlgebra::distributed::Vector<Number> velocity;
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

#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
  create_grid_and_set_boundary_ids(triangulation,
                                   param.h_refinements,
                                   periodic_faces,
                                   SUBDIVISIONS_MESH);
#else
  create_grid_and_set_boundary_ids(triangulation, param.h_refinements, periodic_faces);
#endif

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

  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      conv_diff_operator->initialize_dof_vector_velocity(velocity);
      velocity = 1.0;
    }
  }

  conv_diff_operator->setup_operators_and_solver(1.0 /* use a default value of 1.0 */, &velocity);
}

template<int dim, typename Number>
void
Problem<dim, Number>::apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  LinearAlgebra::distributed::Vector<Number> dst, src;

  conv_diff_operator->initialize_dof_vector(src);
  src = 1.0;
  conv_diff_operator->initialize_dof_vector(dst);

  LinearAlgebra::distributed::Vector<Number> const * velocity_ptr = nullptr;

  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      velocity_ptr = &velocity;
    }
  }

  if(OPERATOR == Operatortype::ConvectiveOperator)
    conv_diff_operator->update_convective_term(1.0 /* time */, velocity_ptr);
  else if(OPERATOR == Operatortype::MassConvectionDiffusionOperator)
    conv_diff_operator->update_conv_diff_operator(1.0 /* time */,
                                                  1.0 /* scaling_factor_mass_matrix */,
                                                  velocity_ptr);

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

      if(OPERATOR == Operatortype::MassOperator)
        conv_diff_operator->apply_mass_matrix(dst, src);
      else if(OPERATOR == Operatortype::ConvectiveOperator)
        conv_diff_operator->apply_convective_term(dst, src);
      else if(OPERATOR == Operatortype::DiffusiveOperator)
        conv_diff_operator->apply_diffusive_term(dst, src);
      else if(OPERATOR == Operatortype::MassConvectionDiffusionOperator)
        conv_diff_operator->apply_conv_diff_operator(dst, src);

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

  WALL_TIMES.push_back(std::tuple<unsigned int, types::global_dof_index, double>(
    param.degree, dofs, dofs_per_walltime));

  pcout << std::endl << " ... done." << std::endl << std::endl;
}

void
do_run(ConvDiff::InputParameters const & param)
{
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

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    // set parameters
    ConvDiff::InputParameters param;
    set_input_parameters(param);

    // p-refinement
    if(RUN_TYPE == RunType::RefineHAndP)
    {
      for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
      {
        // reset degree
        param.degree = degree;

        // h-refinement
        for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
            ++h_refinements)
        {
          // reset h-refinements
          param.h_refinements = h_refinements; // REFINE_LEVELS[degree - 1];

          do_run(param);
        }
      }
    }
#ifdef ENABLE_SUBDIVIDED_HYPERCUBE
    else if(RUN_TYPE == RunType::FixedProblemSize || RUN_TYPE == RunType::IncreasingProblemSize)
    {
      // a vector storing tuples of the form (degree k, refine level l, n_subdivisions_1d)
      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> resolutions;

      // fill resolutions vector

      if(RUN_TYPE == RunType::IncreasingProblemSize)
      {
        AssertThrow(
          DEGREE_MIN == DEGREE_MAX,
          ExcMessage(
            "Only a single polynomial degree can be considered for RunType::IncreasingProblemSize"));
      }

      // k-refinement
      for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
      {
        unsigned int const dim              = double(param.dim);
        double const       dofs_per_element = std::pow(degree + 1, dim);

        fill_resolutions_vector(
          resolutions, degree, dim, dofs_per_element, N_DOFS_MIN, N_DOFS_MAX, RUN_TYPE);
      }

      // loop over resolutions vector and run simulations
      for(auto iter = resolutions.begin(); iter != resolutions.end(); ++iter)
      {
        param.degree        = std::get<0>(*iter);
        param.h_refinements = std::get<1>(*iter);
        SUBDIVISIONS_MESH   = std::get<2>(*iter);

        do_run(param);
      }
    }
#endif
    else
    {
      AssertThrow(false,
                  ExcMessage("Not implemented. Make sure to activate ENABLE_SUBDIVIDED_HYPERCUBE "
                             "for RunType::FixedProblemSize or RunType::IncreasingProblemSize."));
    }

    print_throughput(WALL_TIMES, enum_to_string(OPERATOR));
    WALL_TIMES.clear();
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
