/*
 * unsteady_navier_stokes.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_pressure_correction.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem that has to be solved

// 2D Stokes flow
//#include "incompressible_navier_stokes_test_cases/stokes_guermond.h"
//#include "incompressible_navier_stokes_test_cases/stokes_shahbazi.h"
//#include "incompressible_navier_stokes_test_cases/stokes_curl_flow.h"

// 2D Navier-Stokes flow
//#include "incompressible_navier_stokes_test_cases/couette.h"
//#include "incompressible_navier_stokes_test_cases/poiseuille.h"
//#include "incompressible_navier_stokes_test_cases/cavity.h"
//#include "incompressible_navier_stokes_test_cases/kovasznay.h"
#include "incompressible_navier_stokes_test_cases/vortex.h"
//#include "incompressible_navier_stokes_test_cases/taylor_vortex.h"
//#include "incompressible_navier_stokes_test_cases/tum.h"
//#include "incompressible_navier_stokes_test_cases/orr_sommerfeld.h"
//#include "incompressible_navier_stokes_test_cases/kelvin_helmholtz.h"

// 2D/3D Navier-Stokes flow
//#include "incompressible_navier_stokes_test_cases/flow_past_cylinder.h"

// 3D Navier-Stokes flow
//#include "incompressible_navier_stokes_test_cases/beltrami.h"
//#include "incompressible_navier_stokes_test_cases/unstable_beltrami.h"
//#include "incompressible_navier_stokes_test_cases/cavity_3D.h"
//#include "incompressible_navier_stokes_test_cases/3D_taylor_green_vortex.h"
//#include "incompressible_navier_stokes_test_cases/turbulent_channel.h"
//#include "incompressible_navier_stokes_test_cases/fda_nozzle_benchmark.h"

template<int dim, int degree_u, int degree_p = degree_u - 1, typename Number = double>
class NavierStokesProblem
{
public:
  NavierStokesProblem(unsigned int const refine_steps_space,
                      unsigned int const refine_steps_time = 0);

  void
  solve_problem(bool const do_restart);

private:
  void
  print_header();

  void
  setup_navier_stokes_operation();

  void
  setup_time_integrator(bool const do_restart);

  void
  setup_solvers();

  void
  run_timeloop();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  const unsigned int n_refine_space;

  std::shared_ptr<FieldFunctions<dim>>      field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;
  std::shared_ptr<AnalyticalSolution<dim>>  analytical_solution;

  InputParameters<dim> param;

  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> DGBase;

  typedef DGNavierStokesCoupled<dim, degree_u, degree_p, Number> DGCoupled;

  typedef DGNavierStokesDualSplitting<dim, degree_u, degree_p, Number> DGDualSplitting;

  typedef DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation;

  std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

  std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting;

  std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction;

  typedef PostProcessorBase<dim, degree_u, degree_p, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  typedef TimeIntBDFCoupled<dim, degree_u, Number, DGCoupled> TimeIntCoupled;

  typedef TimeIntBDFDualSplitting<dim, degree_u, Number, DGDualSplitting> TimeIntDualSplitting;

  typedef TimeIntBDFPressureCorrection<dim, degree_u, Number, DGPressureCorrection>
    TimeIntPressureCorrection;

  std::shared_ptr<TimeIntCoupled> time_integrator_coupled;

  std::shared_ptr<TimeIntDualSplitting> time_integrator_dual_splitting;

  std::shared_ptr<TimeIntPressureCorrection> time_integrator_pressure_correction;
};

template<int dim, int degree_u, int degree_p, typename Number>
NavierStokesProblem<dim, degree_u, degree_p, Number>::NavierStokesProblem(
  unsigned int const refine_steps_space,
  unsigned int const refine_steps_time)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space(refine_steps_space)
{
  print_header();
  print_MPI_info(pcout);

  param.set_input_parameters();
  param.check_input_parameters();

  if(param.print_input_parameters == true)
    param.print(pcout);

  field_functions.reset(new FieldFunctions<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions(field_functions);

  analytical_solution.reset(new AnalyticalSolution<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_analytical_solution(analytical_solution);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  bool use_adaptive_time_stepping = false;
  if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
    use_adaptive_time_stepping = true;

  AssertThrow(param.solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor = construct_postprocessor<dim, degree_u, degree_p, Number>(param);

  // initialize navier_stokes_operation
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled.reset(new DGCoupled(triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_coupled;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting.reset(
      new DGDualSplitting(triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_dual_splitting;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction.reset(
      new DGPressureCorrection(triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }


  // initialize time integrator that depends on navier_stokes_operation
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled.reset(new TimeIntCoupled(
      navier_stokes_operation_coupled, param, refine_steps_time, use_adaptive_time_stepping));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting.reset(
      new TimeIntDualSplitting(navier_stokes_operation_dual_splitting,
                               param,
                               refine_steps_time,
                               use_adaptive_time_stepping));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction,
                                    param,
                                    refine_steps_time,
                                    use_adaptive_time_stepping));
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
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                     based on a matrix-free implementation                       " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_navier_stokes_operation()
{
  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions,
                                 analytical_solution);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_time_integrator(bool const do_restart)
{
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled->setup(do_restart);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting->setup(do_restart);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction->setup(do_restart);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_solvers()
{
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled->setup_solvers(
      time_integrator_coupled->get_scaling_factor_time_derivative_term());
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting->setup_solvers(
      time_integrator_dual_splitting->get_time_step_size(),
      time_integrator_dual_splitting->get_scaling_factor_time_derivative_term());
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction->setup_solvers(
      time_integrator_pressure_correction->get_time_step_size(),
      time_integrator_pressure_correction->get_scaling_factor_time_derivative_term());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::run_timeloop()
{
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    if(this->param.problem_type == ProblemType::Steady)
      time_integrator_coupled->timeloop_steady_problem();
    else if(this->param.problem_type == ProblemType::Unsteady)
      time_integrator_coupled->timeloop();
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(this->param.problem_type == ProblemType::Steady)
      time_integrator_dual_splitting->timeloop_steady_problem();
    else if(this->param.problem_type == ProblemType::Unsteady)
      time_integrator_dual_splitting->timeloop();
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    if(this->param.problem_type == ProblemType::Steady)
      time_integrator_pressure_correction->timeloop_steady_problem();
    else if(this->param.problem_type == ProblemType::Unsteady)
      time_integrator_pressure_correction->timeloop();
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::solve_problem(bool const do_restart)
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions(triangulation,
                                          n_refine_space,
                                          boundary_descriptor_velocity,
                                          boundary_descriptor_pressure,
                                          periodic_faces);

  print_grid_data(pcout, n_refine_space, triangulation);

  setup_navier_stokes_operation();

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  setup_time_integrator(do_restart);

  setup_solvers();

  run_timeloop();
}

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

    bool do_restart = false;
    if(argc > 1)
    {
      do_restart = std::atoi(argv[1]);
      if(do_restart)
      {
        AssertThrow(REFINE_STEPS_SPACE_MIN == REFINE_STEPS_SPACE_MAX,
                    ExcMessage("Spatial refinement with restart not possible!"));

        // this does in principle work although it doesn't make much sense
        if(REFINE_STEPS_TIME_MIN != REFINE_STEPS_TIME_MAX &&
           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Warning: you are starting from a restart and refine the time steps!"
                    << std::endl;
      }
    }

    // mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX;
        ++refine_steps_space)
    {
      // time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;
          refine_steps_time <= REFINE_STEPS_TIME_MAX;
          ++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, VALUE_TYPE>
          navier_stokes_problem(refine_steps_space, refine_steps_time);

        navier_stokes_problem.solve_problem(do_restart);
      }
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
