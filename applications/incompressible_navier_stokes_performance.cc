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
#include <deal.II/grid/grid_tools.h>

// timer
#include <deal.II/base/timer.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

#include "../include/postprocessor/output_data.h"

// Navier-Stokes operator
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem to be used for throughput measurements

#include "incompressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"

/**************************************************************************************/
/*                                                                                    */
/*                          FURTHER INPUT PARAMETERS                                  */
/*                                                                                    */
/**************************************************************************************/

// set the polynomial degree k of the shape functions
unsigned int const FE_DEGREE_U_MIN = 2;
unsigned int const FE_DEGREE_U_MAX = 2;

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

Operator OPERATOR = Operator::ConvectiveOperator;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 10;  // take the minimum of outer repetitions

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double>> wall_times;


template<int dim, int degree_u, int degree_p, typename Number>
class NavierStokesProblem
{
public:
  NavierStokesProblem(unsigned int const refine_steps_space);

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

  unsigned int const n_refine_space;

  std::shared_ptr<FieldFunctions<dim>>      field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolution<dim>> analytical_solution;

  InputParameters<dim> param;

  typedef PostProcessorBase<dim, degree_u, degree_p, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> DGBase;

  typedef DGNavierStokesCoupled<dim, degree_u, degree_p, Number> DGCoupled;

  typedef DGNavierStokesDualSplitting<dim, degree_u, degree_p, Number> DGDualSplitting;

  typedef DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation;

  std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

  std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting;

  std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;
};

template<int dim, int degree_u, int degree_p, typename Number>
NavierStokesProblem<dim, degree_u, degree_p, Number>::NavierStokesProblem(
  unsigned int const refine_steps_space)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_refine_space(refine_steps_space),
    n_repetitions_inner(N_REPETITIONS_INNER),
    n_repetitions_outer(N_REPETITIONS_OUTER)
{
  print_header();

  // input parameters
  param.set_input_parameters();
  param.check_input_parameters();

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

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  postprocessor = construct_postprocessor<dim, degree_u, degree_p, Number>(param);

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

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::print_header()
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

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup()
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_ids(triangulation, n_refine_space, periodic_faces);

  print_grid_data(pcout, n_refine_space, *triangulation);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);

  // setup Navier-Stokes operation
  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));
  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions,
                                 analytical_solution);

  // setup Navier-Stokes solvers
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled->setup_solvers(1.0);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting->setup_solvers(1.0);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction->setup_solvers(1.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // Vectors needed for coupled solution approach
  LinearAlgebra::distributed::BlockVector<Number> dst1, src1;

  // ... for dual splitting, pressure-correction.
  LinearAlgebra::distributed::Vector<Number> dst2, src2;

  // initialize vectors
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled->initialize_block_vector_velocity_pressure(dst1);
    navier_stokes_operation_coupled->initialize_block_vector_velocity_pressure(src1);
    src1 = 1.0;

    // set linearized solution -> required to evaluate the linearized operator
    navier_stokes_operation_coupled->set_solution_linearization(src1);
    // set sum_alphai_ui -> required to evaluate the nonlinear residual
    // in case an unsteady problem is considered
    navier_stokes_operation_coupled->set_sum_alphai_ui(&src1.block(0));

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
    if(OPERATOR == Operator::ConvectiveOperator || OPERATOR == Operator::VelocityConvDiffOperator ||
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
          navier_stokes_operation_coupled->evaluate_nonlinear_residual(dst1,src1);
        else if(OPERATOR == Operator::CoupledLinearized)
          navier_stokes_operation_coupled->vmult(dst1,src1);
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
          navier_stokes_operation_pressure_correction->apply_momentum_operator(dst2,src2,src2);
        else if(OPERATOR == Operator::ConvectiveOperator)
          navier_stokes_operation_dual_splitting->evaluate_convective_term(dst2,src2,0.0);
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
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducable results."
      << std::endl;
  }

  unsigned int dofs      = 0;
  unsigned int fe_degree = 1;

  if(OPERATOR == Operator::CoupledNonlinearResidual || OPERATOR == Operator::CoupledLinearized)
  {
    dofs = navier_stokes_operation->get_dof_handler_u().n_dofs() +
           navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = degree_u;
  }
  else if(OPERATOR == Operator::ConvectiveOperator ||
          OPERATOR == Operator::VelocityConvDiffOperator ||
          OPERATOR == Operator::HelmholtzOperator || OPERATOR == Operator::ProjectionOperator ||
          OPERATOR == Operator::InverseMassMatrix)
  {
    dofs = navier_stokes_operation->get_dof_handler_u().n_dofs();

    fe_degree = degree_u;
  }
  else if(OPERATOR == Operator::PressurePoissonOperator)
  {
    dofs = navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = degree_p;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double wall_time_per_dofs = wall_time / (double)dofs;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4) << "t_wall/DoF [s]:  " << wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/sec:        " << 1./wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/(sec*core): " << 1./wall_time_per_dofs/(double)N_mpi_processes << std::endl;
  // clang-format on

  wall_times.push_back(std::pair<unsigned int, double>(fe_degree, wall_time_per_dofs));


  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    // reset sum_alphai_ui
    navier_stokes_operation_coupled->set_sum_alphai_ui(nullptr);
  }


  pcout << std::endl << " ... done." << std::endl << std::endl;
}

void
print_wall_times(std::vector<std::pair<unsigned int, double>> const & wall_times,
                 unsigned int const                                   refine_steps_space)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::string str_operator_type[] = {"CoupledNonlinearResidual",
                                       "CoupledLinearized",
                                       "PressurePoissonOperator",
                                       "ConvectiveOperator",
                                       "HelmholtzOperator",
                                       "ProjectionOperator",
                                       "VelocityConvDiffOperator",
                                       "InverseMassMatrix"};

    // clang-format off
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Operator type: " << str_operator_type[(int)OPERATOR]
              << std::endl << std::endl
              << "Wall times for refine level l = " << refine_steps_space << ":"
              << std::endl << std::endl
              << "  k    " << "t_wall/DoF [s] " << "DoFs/sec   " << "DoFs/(sec*core) " << std::endl;

    typedef typename std::vector<std::pair<unsigned int, double> >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << "  " << std::setw(5) << std::left << it->first
                << std::setw(2) << std::left << std::scientific << std::setprecision(4) << it->second
                << "     " << std::setw(2) << std::left << std::scientific << std::setprecision(4) << 1./it->second
                << " " << std::setw(2) << std::left << std::scientific << std::setprecision(4) << 1./it->second/(double)N_mpi_processes
                << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
    // clang-format on
  }
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
template<int dim, int degree_u, int max_degree_u, typename Number>
class NavierStokesPrecompiled
{
public:
  static void
  run()
  {
    NavierStokesPrecompiled<dim, degree_u, degree_u, Number>::run();

    NavierStokesPrecompiled<dim, degree_u + 1, max_degree_u, Number>::run();
  }
};

/*
 * specialization of templates: degree_u == max_degree_u
 * Note that degree_p = degree_u - 1.
 */
template<int dim, int degree_u, typename Number>
class NavierStokesPrecompiled<dim, degree_u, degree_u, Number>
{
public:
  static void
  run()
  {
    typedef NavierStokesProblem<dim, degree_u, degree_u - 1 /* degree_p*/, Number>
      NAVIER_STOKES_PROBLEM;

    NAVIER_STOKES_PROBLEM navier_stokes_problem(REFINE_LEVELS[degree_u - 1]);
    navier_stokes_problem.setup();
    navier_stokes_problem.apply_operator();
  }
};

int
main(int argc, char ** argv)
{
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

    // mesh refinements
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX;
        ++refine_steps_space)
    {
      // increasing polynomial degrees
      typedef NavierStokesPrecompiled<DIMENSION, FE_DEGREE_U_MIN, FE_DEGREE_U_MAX, VALUE_TYPE>
        NAVIER_STOKES;

      NAVIER_STOKES::run();

      print_wall_times(wall_times, refine_steps_space);
      wall_times.clear();
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
