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
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_pressure_correction.h"

using namespace dealii;


// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/couette.h"
//#include "incompressible_navier_stokes_test_cases/poiseuille.h"
//#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/stokes_guermond.h"
//#include "incompressible_navier_stokes_test_cases/stokes_shahbazi.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
//#include "incompressible_navier_stokes_test_cases/vortex.h"
//#include "incompressible_navier_stokes_test_cases/taylor_vortex.h"
#include "incompressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"
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
unsigned int const FE_DEGREE_U_MAX = FE_DEGREE_VELOCITY+1;

// Select the operator to be applied
enum class OperatorType{
  CoupledNonlinearResidual, // nonlinear residual of coupled system of equations
  CoupledLinearized,        // linearized system of equations for coupled solution approach
  PressurePoissonOperator,  // negative Laplace operator (scalar quantity, pressure)
  HelmholtzOperator,        // mass + viscous (vectorial quantity, velocity)
  ProjectionOperator,       // mass + divergence penalty + continuity penalty (vectorial quantity, velocity)
  VelocityConvDiffOperator, // mass + convective + viscous (vectorial quantity, velocity)
  InverseMassMatrix         // inverse mass matrix operator (vectorial quantity, velocity)
};

OperatorType OPERATOR_TYPE = OperatorType::InverseMassMatrix; // CoupledLinearized;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS_INNER = 100; // take the average of inner repetitions
unsigned int const N_REPETITIONS_OUTER = 5;   // take the minimum of outer repetitions

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
  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure;

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  InputParametersNavierStokes<dim> param;

  std::shared_ptr<DGNavierStokesBase<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation;

  std::shared_ptr<DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_coupled;

  std::shared_ptr<DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_dual_splitting;

  std::shared_ptr<DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_pressure_correction;

  // number of matrix-vector products
  unsigned int const n_repetitions_inner, n_repetitions_outer;
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
  n_repetitions_inner(N_REPETITIONS_INNER),
  n_repetitions_outer(N_REPETITIONS_OUTER)
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

  boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokesU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokesP<dim>());

  AssertThrow(param.solver_type == SolverType::Unsteady,
      ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize navier_stokes_operation
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled.reset(
        new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation,param));

    navier_stokes_operation = navier_stokes_operation_coupled;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting.reset(
        new DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation,param));

    navier_stokes_operation = navier_stokes_operation_dual_splitting;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction.reset(
        new DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation,param));

    navier_stokes_operation = navier_stokes_operation_pressure_correction;
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  // check that the operator type is consistent with the solution approach (coupled vs. splitting)
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    AssertThrow(OPERATOR_TYPE == OperatorType::CoupledNonlinearResidual ||
                OPERATOR_TYPE == OperatorType::CoupledLinearized ||
                OPERATOR_TYPE == OperatorType::InverseMassMatrix,
                ExcMessage("Invalid operator specified for coupled solution approach."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    AssertThrow(OPERATOR_TYPE == OperatorType::PressurePoissonOperator ||
                OPERATOR_TYPE == OperatorType::HelmholtzOperator ||
                OPERATOR_TYPE == OperatorType::ProjectionOperator ||
                OPERATOR_TYPE == OperatorType::InverseMassMatrix,
                ExcMessage("Invalid operator specified for dual splitting scheme."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    AssertThrow(OPERATOR_TYPE == OperatorType::PressurePoissonOperator ||
                OPERATOR_TYPE == OperatorType::VelocityConvDiffOperator ||
                OPERATOR_TYPE == OperatorType::ProjectionOperator ||
                OPERATOR_TYPE == OperatorType::InverseMassMatrix,
                ExcMessage("Invalid operator specified for pressure-correction scheme."));
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
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

  // setup Navier-Stokes operation
  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));
  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  // setup Navier-Stokes solvers
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled->setup_solvers(1.0);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting->setup_solvers(1.0,1.0);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction->setup_solvers(1.0,1.0);
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_operator()
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // Vectors needed for coupled solution approach
  parallel::distributed::BlockVector<VALUE_TYPE> dst1, src1;

  // ... for dual splitting, pressure-correction.
  parallel::distributed::Vector<VALUE_TYPE> dst2, src2;

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
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(OPERATOR_TYPE == OperatorType::HelmholtzOperator ||
       OPERATOR_TYPE == OperatorType::ProjectionOperator ||
       OPERATOR_TYPE == OperatorType::InverseMassMatrix)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(dst2);
    }
    else if(OPERATOR_TYPE == OperatorType::PressurePoissonOperator)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false,ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    if(OPERATOR_TYPE == OperatorType::VelocityConvDiffOperator ||
       OPERATOR_TYPE == OperatorType::ProjectionOperator ||
       OPERATOR_TYPE == OperatorType::InverseMassMatrix)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_velocity(dst2);
    }
    else if(OPERATOR_TYPE == OperatorType::PressurePoissonOperator)
    {
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(src2);
      navier_stokes_operation_dual_splitting->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false,ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }


  // Timer and wall times
  Timer timer;
  double wall_time = std::numeric_limits<double>::max();

  for(unsigned int i_outer=0; i_outer<n_repetitions_outer; ++i_outer)
  {
    double current_wall_time = 0.0;

    // apply matrix-vector product several times
    for(unsigned int i=0; i<n_repetitions_inner; ++i)
    {
      timer.restart();

      if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        if(OPERATOR_TYPE == OperatorType::CoupledNonlinearResidual)
          navier_stokes_operation_coupled->evaluate_nonlinear_residual(dst1,src1);
        else if(OPERATOR_TYPE == OperatorType::CoupledLinearized)
          navier_stokes_operation_coupled->vmult(dst1,src1);
        else if(OPERATOR_TYPE == OperatorType::InverseMassMatrix)
          navier_stokes_operation_coupled->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        if(OPERATOR_TYPE == OperatorType::HelmholtzOperator)
          navier_stokes_operation_dual_splitting->apply_helmholtz_operator(dst2,src2);
        else if(OPERATOR_TYPE == OperatorType::ProjectionOperator)
          navier_stokes_operation_dual_splitting->apply_projection_operator(dst2,src2);
        else if(OPERATOR_TYPE == OperatorType::PressurePoissonOperator)
          navier_stokes_operation_dual_splitting->apply_laplace_operator(dst2,src2);
        else if(OPERATOR_TYPE == OperatorType::InverseMassMatrix)
          navier_stokes_operation_dual_splitting->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
      {
        if(OPERATOR_TYPE == OperatorType::VelocityConvDiffOperator)
          navier_stokes_operation_pressure_correction->apply_velocity_conv_diff_operator(dst2,src2,src2);
        else if(OPERATOR_TYPE == OperatorType::ProjectionOperator)
          navier_stokes_operation_pressure_correction->apply_projection_operator(dst2,src2);
        else if(OPERATOR_TYPE == OperatorType::PressurePoissonOperator)
          navier_stokes_operation_pressure_correction->apply_laplace_operator(dst2,src2);
        else if(OPERATOR_TYPE == OperatorType::InverseMassMatrix)
          navier_stokes_operation_pressure_correction->apply_inverse_mass_matrix(dst2,src2);
        else
          AssertThrow(false,ExcMessage("Not implemented."));
      }
      else
      {
        AssertThrow(false,ExcMessage("Not implemented."));
      }

      current_wall_time += timer.wall_time();
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time,current_wall_time);
  }

  if(wall_time*n_repetitions_inner*n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout << std::endl
                << "WARNING: One should use a larger number of matrix-vector products to obtain reproducable results."
                << std::endl;
  }

  unsigned int dofs = 0;
  unsigned int fe_degree = 1;

  if(OPERATOR_TYPE == OperatorType::CoupledNonlinearResidual ||
     OPERATOR_TYPE == OperatorType::CoupledLinearized)
  {
    dofs =   navier_stokes_operation->get_dof_handler_u().n_dofs()
           + navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = fe_degree_u;
  }
  else if(OPERATOR_TYPE == OperatorType::VelocityConvDiffOperator ||
          OPERATOR_TYPE == OperatorType::HelmholtzOperator ||
          OPERATOR_TYPE == OperatorType::ProjectionOperator ||
          OPERATOR_TYPE == OperatorType::InverseMassMatrix)
  {
    dofs = navier_stokes_operation->get_dof_handler_u().n_dofs();

    fe_degree = fe_degree_u;
  }
  else if(OPERATOR_TYPE == OperatorType::PressurePoissonOperator)
  {
    dofs = navier_stokes_operation->get_dof_handler_p().n_dofs();

    fe_degree = fe_degree_p;
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  double wall_time_per_dofs = wall_time / (double) dofs;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  pcout << std::endl
        << std::scientific << std::setprecision(4) << "t_wall/DoF [s]:  " << wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/sec:        " << 1./wall_time_per_dofs << std::endl
        << std::scientific << std::setprecision(4) << "DoFs/(sec*core): " << 1./wall_time_per_dofs/(double)N_mpi_processes << std::endl;

  wall_times.push_back(std::pair<unsigned int,double>(fe_degree,wall_time_per_dofs));


  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    // reset sum_alphai_ui
    navier_stokes_operation_coupled->set_sum_alphai_ui(nullptr);
  }


  pcout << std::endl << " ... done." << std::endl << std::endl;
}

void print_wall_times(std::vector<std::pair<unsigned int, double> > const &wall_times,
                      unsigned int const                                  refine_steps_space)
{
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::string str_operator_type[] = {"CoupledNonlinearResidual",
                                      "CoupledLinearized",
                                      "PressurePoissonOperator",
                                      "HelmholtzOperator",
                                      "ProjectionOperator",
                                      "VelocityConvDiffOperator"};

    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Operator type: " << str_operator_type[(int)OPERATOR_TYPE] << std::endl << std::endl
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

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    //mesh refinements
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX; ++refine_steps_space)
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



