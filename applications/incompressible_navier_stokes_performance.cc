/*
 * incompressible_navier_stokes_performance.cc
 *
 *  Created on: May 5, 2017
 *      Author: fehn
 */

// deal.ii

#include <deal.II/base/revision.h>

// triangulation
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

// timer
#include <deal.II/base/timer.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor_base.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/mesh_resolution_generator_hypercube.h"
#include "../include/functionalities/print_general_infos.h"
#include "../include/functionalities/print_throughput.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem to be used for throughput measurements

#include "incompressible_navier_stokes_test_cases/periodic_box.h"

RunType const RUN_TYPE = RunType::FixedProblemSize; // IncreasingProblemSize;

/*
 * Specify minimum and maximum problem size for
 *  RunType::FixedProblemSize
 *  RunType::IncreasingProblemSize
 */
types::global_dof_index N_DOFS_MIN = 5e4;
types::global_dof_index N_DOFS_MAX = 2e5;

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

// Note: Make sure that the correct time integration scheme is selected in the input file that is
//       compatible with the Operator type specified here. This also includes the treatment of the
//       convective term (explicit/implicit), e.g., specifying VelocityConvDiffOperator together
//       with an explicit treatment of the convective term will only apply the Helmholtz-like
//       operator.

// clang-format off
enum class Operator{
  CoupledNonlinearResidual, // nonlinear residual of coupled system of equations
  CoupledLinearized,        // linearized system of equations for coupled solution approach
  PressurePoissonOperator,  // negative Laplace operator (scalar quantity, pressure)
  ConvectiveOperator,       // convective term (vectorial quantity, velocity)
  HelmholtzOperator,        // mass + viscous (vectorial quantity, velocity)
  ProjectionOperator,       // mass + divergence penalty + continuity penalty (vectorial quantity, velocity)
  VelocityConvDiffOperator, // mass + convective + viscous (vectorial quantity, velocity)
  InverseMassMatrix         // inverse mass matrix operator (vectorial quantity, velocity)
};
// clang-format on

Operator OPERATOR = Operator::VelocityConvDiffOperator;

std::string
enum_to_string(Operator const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operator::CoupledNonlinearResidual: string_type = "CoupledNonlinearResidual"; break;
    case Operator::CoupledLinearized:        string_type = "CoupledLinearized";        break;
    case Operator::PressurePoissonOperator:  string_type = "PressurePoissonOperator";  break;
    case Operator::ConvectiveOperator:       string_type = "ConvectiveOperator";       break;
    case Operator::HelmholtzOperator:        string_type = "HelmholtzOperator";        break;
    case Operator::ProjectionOperator:       string_type = "ProjectionOperator";       break;
    case Operator::VelocityConvDiffOperator: string_type = "VelocityConvDiffOperator"; break;
    case Operator::InverseMassMatrix:        string_type = "InverseMassMatrix";        break;

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

template<int dim, typename Number>
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

  std::shared_ptr<FieldFunctions<dim>>      field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;

  InputParameters param;

  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  typedef DGNavierStokesBase<dim, Number>               DGBase;
  typedef DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation;

  std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

  std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting;

  std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;

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
  << "                High-order discontinuous Galerkin discretization                 " << std::endl
  << "                  of the incompressible Navier-Stokes equations                  " << std::endl
  << "                      based on a matrix-free implementation                      " << std::endl
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

  // input parameters
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

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor = construct_postprocessor<dim, Number>(param);

  AssertThrow(param.solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize navier_stokes_operation
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled.reset(new DGCoupled(*triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_coupled;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting.reset(
      new DGDualSplitting(*triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_dual_splitting;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction.reset(
      new DGPressureCorrection(*triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));
  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  navier_stokes_operation->initialize_vector_velocity(velocity);
  velocity = 1.0;
  navier_stokes_operation->setup_solvers(1.0 /* dummy */, velocity);


  // check that the operator type is consistent with the solution approach (coupled vs. splitting)
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    AssertThrow(OPERATOR == Operator::ConvectiveOperator ||
                  OPERATOR == Operator::CoupledNonlinearResidual ||
                  OPERATOR == Operator::CoupledLinearized ||
                  OPERATOR == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for coupled solution approach."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    AssertThrow(OPERATOR == Operator::ConvectiveOperator ||
                  OPERATOR == Operator::PressurePoissonOperator ||
                  OPERATOR == Operator::HelmholtzOperator ||
                  OPERATOR == Operator::ProjectionOperator ||
                  OPERATOR == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for dual splitting scheme."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    AssertThrow(OPERATOR == Operator::ConvectiveOperator ||
                  OPERATOR == Operator::PressurePoissonOperator ||
                  OPERATOR == Operator::VelocityConvDiffOperator ||
                  OPERATOR == Operator::ProjectionOperator ||
                  OPERATOR == Operator::InverseMassMatrix,
                ExcMessage("Invalid operator specified for pressure-correction scheme."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // Vectors needed for coupled solution approach
  LinearAlgebra::distributed::BlockVector<Number> dst1, src1;

  // ... for dual splitting, pressure-correction.
  LinearAlgebra::distributed::Vector<Number> dst2, src2;

  // set velocity required for evaluation of linearized operators
  navier_stokes_operation->set_velocity_ptr(velocity);

  // initialize vectors
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled->initialize_block_vector_velocity_pressure(dst1);
    navier_stokes_operation_coupled->initialize_block_vector_velocity_pressure(src1);
    src1 = 1.0;

    if(OPERATOR == Operator::ConvectiveOperator || OPERATOR == Operator::InverseMassMatrix)
    {
      navier_stokes_operation_coupled->initialize_vector_velocity(src2);
      navier_stokes_operation_coupled->initialize_vector_velocity(dst2);
    }
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(OPERATOR == Operator::ConvectiveOperator || OPERATOR == Operator::HelmholtzOperator ||
       OPERATOR == Operator::ProjectionOperator || OPERATOR == Operator::InverseMassMatrix)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(dst2);
    }
    else if(OPERATOR == Operator::PressurePoissonOperator)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    if(OPERATOR == Operator::VelocityConvDiffOperator || OPERATOR == Operator::ProjectionOperator ||
       OPERATOR == Operator::InverseMassMatrix)
    {
      navier_stokes_operation_pressure_correction->initialize_vector_velocity(src2);
      navier_stokes_operation_pressure_correction->initialize_vector_velocity(dst2);
    }
    else if(OPERATOR == Operator::PressurePoissonOperator)
    {
      navier_stokes_operation_pressure_correction->initialize_vector_pressure(src2);
      navier_stokes_operation_pressure_correction->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
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

      // clang-format off
      if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        if(OPERATOR == Operator::CoupledNonlinearResidual)
          navier_stokes_operation_coupled->evaluate_nonlinear_residual(dst1,src1,&src1.block(0), 0.0, 1.0);
        else if(OPERATOR == Operator::CoupledLinearized)
          navier_stokes_operation_coupled->apply_linearized_problem(dst1,src1, 0.0, 1.0);
        else if(OPERATOR == Operator::ConvectiveOperator)
          navier_stokes_operation_coupled->evaluate_convective_term(dst2,src2,0.0);
        else if(OPERATOR == Operator::InverseMassMatrix)
          navier_stokes_operation_coupled->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        if(OPERATOR == Operator::HelmholtzOperator)
          navier_stokes_operation_dual_splitting->apply_helmholtz_operator(dst2,src2);
        else if(OPERATOR == Operator::ConvectiveOperator)
          navier_stokes_operation_dual_splitting->evaluate_convective_term(dst2,src2,0.0);
        else if(OPERATOR == Operator::ProjectionOperator)
          navier_stokes_operation_dual_splitting->apply_projection_operator(dst2,src2);
        else if(OPERATOR == Operator::PressurePoissonOperator)
          navier_stokes_operation_dual_splitting->apply_laplace_operator(dst2,src2);
        else if(OPERATOR == Operator::InverseMassMatrix)
          navier_stokes_operation_dual_splitting->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
      {
        if(OPERATOR == Operator::VelocityConvDiffOperator)
          navier_stokes_operation_pressure_correction->apply_momentum_operator(dst2,src2);
        else if(OPERATOR == Operator::ProjectionOperator)
          navier_stokes_operation_pressure_correction->apply_projection_operator(dst2,src2);
        else if(OPERATOR == Operator::PressurePoissonOperator)
          navier_stokes_operation_pressure_correction->apply_laplace_operator(dst2,src2);
        else if(OPERATOR == Operator::InverseMassMatrix)
          navier_stokes_operation_pressure_correction->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else
      {
        AssertThrow(false,ExcMessage("Not implemented."));
      }
      // clang-format on

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

  types::global_dof_index dofs      = 0;
  unsigned int            fe_degree = 1;

  if(OPERATOR == Operator::CoupledNonlinearResidual || OPERATOR == Operator::CoupledLinearized)
  {
    dofs = navier_stokes_operation->get_dof_handler_u().n_dofs() +
           navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = param.degree_u;
  }
  else if(OPERATOR == Operator::ConvectiveOperator ||
          OPERATOR == Operator::VelocityConvDiffOperator ||
          OPERATOR == Operator::HelmholtzOperator || OPERATOR == Operator::ProjectionOperator ||
          OPERATOR == Operator::InverseMassMatrix)
  {
    dofs = navier_stokes_operation->get_dof_handler_u().n_dofs();

    fe_degree = param.degree_u;
  }
  else if(OPERATOR == Operator::PressurePoissonOperator)
  {
    dofs = navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = param.get_degree_p();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  WALL_TIMES.push_back(
    std::tuple<unsigned int, types::global_dof_index, double>(fe_degree, dofs, dofs_per_walltime));

  pcout << std::endl << " ... done." << std::endl << std::endl;
}

unsigned int
get_dofs_per_element(InputParameters const & param)
{
  unsigned int const dim                       = param.dim;
  unsigned int const velocity_dofs_per_element = dim * std::pow(param.degree_u + 1, dim);
  unsigned int const pressure_dofs_per_element = std::pow(param.get_degree_p() + 1, dim);

  if(OPERATOR == Operator::CoupledNonlinearResidual || OPERATOR == Operator::CoupledLinearized)
  {
    return velocity_dofs_per_element + pressure_dofs_per_element;
  }
  // velocity
  else if(OPERATOR == Operator::ConvectiveOperator ||
          OPERATOR == Operator::VelocityConvDiffOperator ||
          OPERATOR == Operator::HelmholtzOperator || OPERATOR == Operator::ProjectionOperator ||
          OPERATOR == Operator::InverseMassMatrix)
  {
    return velocity_dofs_per_element;
  }
  // pressure
  else if(OPERATOR == Operator::PressurePoissonOperator)
  {
    return pressure_dofs_per_element;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

void
do_run(InputParameters const & param)
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

    InputParameters param;
    set_input_parameters(param);

    if(RUN_TYPE == RunType::RefineHAndP)
    {
      // p-refinement
      for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
      {
        // reset degree
        param.degree_u = degree;

        // h-refinement
        for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
            ++h_refinements)
        {
          // reset h-refinements
          param.h_refinements = h_refinements;

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
        unsigned int const dim = double(param.dim);
        param.degree_u         = degree; // reset degree to calculate the correct dofs_per_element
        double const dofs_per_element = get_dofs_per_element(param);

        fill_resolutions_vector(
          resolutions, degree, dim, dofs_per_element, N_DOFS_MIN, N_DOFS_MAX, RUN_TYPE);
      }

      // loop over resolutions vector and run simulations
      for(auto iter = resolutions.begin(); iter != resolutions.end(); ++iter)
      {
        param.degree_u      = std::get<0>(*iter);
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
