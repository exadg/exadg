/*
 *unsteady_navier_stokes_two_domains.cc
 *
 *  Created on: 2017
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

#include "../include/incompressible_navier_stokes/interface_space_time/operator.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Paramters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/turbulent_channel_two_domains.h"
//#include "incompressible_navier_stokes_test_cases/backward_facing_step_two_domains.h"
#include "incompressible_navier_stokes_test_cases/fda_nozzle_benchmark.h"

template<int dim, int degree_u, int degree_p = degree_u - 1, typename Number = double>
class NavierStokesProblem
{
public:
  NavierStokesProblem(unsigned int const refine_steps_space1,
                      unsigned int const refine_steps_space2,
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
  set_start_time();

  void
  synchronize_time_step_size();

  void
  setup_solvers();

  void
  run_timeloop();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation_1, triangulation_2;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces_1, periodic_faces_2;

  const unsigned int n_refine_space_domain1, n_refine_space_domain2;

  bool use_adaptive_time_stepping;

  std::shared_ptr<FieldFunctions<dim>>      field_functions_1, field_functions_2;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity_1,
    boundary_descriptor_velocity_2;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure_1,
    boundary_descriptor_pressure_2;

  std::shared_ptr<AnalyticalSolution<dim>> analytical_solution_1, analytical_solution_2;

  InputParameters<dim> param_1, param_2;

  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> DGBase;

  typedef DGNavierStokesCoupled<dim, degree_u, degree_p, Number> DGCoupled;

  typedef DGNavierStokesDualSplitting<dim, degree_u, degree_p, Number> DGDualSplitting;

  typedef DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation_1, navier_stokes_operation_2;

  std::shared_ptr<DGCoupled> navier_stokes_operation_coupled_1, navier_stokes_operation_coupled_2;

  std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting_1,
    navier_stokes_operation_dual_splitting_2;

  std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction_1,
    navier_stokes_operation_pressure_correction_2;

  typedef PostProcessorBase<dim, degree_u, degree_p, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor_1, postprocessor_2;

  typedef TimeIntBDF<dim, Number>                   TimeInt;
  typedef TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> time_integrator_1, time_integrator_2;
};

template<int dim, int degree_u, int degree_p, typename Number>
NavierStokesProblem<dim, degree_u, degree_p, Number>::NavierStokesProblem(
  unsigned int const refine_steps_space1,
  unsigned int const refine_steps_space2,
  unsigned int const refine_steps_time)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    triangulation_1(MPI_COMM_WORLD,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    triangulation_2(MPI_COMM_WORLD,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    n_refine_space_domain1(refine_steps_space1),
    n_refine_space_domain2(refine_steps_space2),
    use_adaptive_time_stepping(false)
{
  param_1.set_input_parameters(1);
  param_1.check_input_parameters();

  param_2.set_input_parameters(2);
  param_2.check_input_parameters();

  print_header();
  print_MPI_info(pcout);
  if(param_1.print_input_parameters == true)
  {
    pcout << std::endl << "List of input parameters for DOMAIN 1:" << std::endl;
    param_1.print(pcout);
  }
  if(param_2.print_input_parameters == true)
  {
    pcout << std::endl << "List of input parameters for DOMAIN 2:" << std::endl;
    param_2.print(pcout);
  }

  field_functions_1.reset(new FieldFunctions<dim>());
  field_functions_2.reset(new FieldFunctions<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions_1(field_functions_1);
  set_field_functions_2(field_functions_2);

  analytical_solution_1.reset(new AnalyticalSolution<dim>());
  analytical_solution_2.reset(new AnalyticalSolution<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_analytical_solution(analytical_solution_1);
  set_analytical_solution(analytical_solution_2);

  boundary_descriptor_velocity_1.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_1.reset(new BoundaryDescriptorP<dim>());

  boundary_descriptor_velocity_2.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_2.reset(new BoundaryDescriptorP<dim>());

  // constant vs. adaptive time stepping
  AssertThrow(param_1.calculation_of_time_step_size == param_2.calculation_of_time_step_size,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  if(param_1.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    use_adaptive_time_stepping = true;
  }

  AssertThrow(param_1.solver_type == SolverType::Unsteady &&
                param_2.solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor_1 = construct_postprocessor<dim, degree_u, degree_p, Number>(param_1);
  postprocessor_2 = construct_postprocessor<dim, degree_u, degree_p, Number>(param_2);

  // initialize navier_stokes_operation_1 (DOMAIN 1)
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_1.reset(
      new DGCoupled(triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_coupled_1;

    time_integrator_1.reset(new TimeIntCoupled(navier_stokes_operation_coupled_1,
                                               navier_stokes_operation_coupled_1,
                                               param_1,
                                               refine_steps_time,
                                               use_adaptive_time_stepping));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_1.reset(
      new DGDualSplitting(triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_dual_splitting_1;

    time_integrator_1.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting_1,
                                                     navier_stokes_operation_dual_splitting_1,
                                                     param_1,
                                                     refine_steps_time,
                                                     use_adaptive_time_stepping));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_1.reset(
      new DGPressureCorrection(triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_pressure_correction_1;

    time_integrator_1.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction_1,
                                    navier_stokes_operation_pressure_correction_1,
                                    param_1,
                                    refine_steps_time,
                                    use_adaptive_time_stepping));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize navier_stokes_operation_2 (DOMAIN 2)
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_2.reset(
      new DGCoupled(triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_coupled_2;

    time_integrator_2.reset(new TimeIntCoupled(navier_stokes_operation_coupled_2,
                                               navier_stokes_operation_coupled_2,
                                               param_2,
                                               refine_steps_time,
                                               use_adaptive_time_stepping));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_2.reset(
      new DGDualSplitting(triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_dual_splitting_2;

    time_integrator_2.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting_2,
                                                     navier_stokes_operation_dual_splitting_2,
                                                     param_2,
                                                     refine_steps_time,
                                                     use_adaptive_time_stepping));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_2.reset(
      new DGPressureCorrection(triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_pressure_correction_2;

    time_integrator_2.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction_2,
                                    navier_stokes_operation_pressure_correction_2,
                                    param_2,
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
  AssertThrow(navier_stokes_operation_1.get() != 0, ExcMessage("Not initialized."));
  AssertThrow(navier_stokes_operation_2.get() != 0, ExcMessage("Not initialized."));

  navier_stokes_operation_1->setup(periodic_faces_1,
                                   boundary_descriptor_velocity_1,
                                   boundary_descriptor_pressure_1,
                                   field_functions_1,
                                   analytical_solution_1);

  navier_stokes_operation_2->setup(periodic_faces_2,
                                   boundary_descriptor_velocity_2,
                                   boundary_descriptor_pressure_2,
                                   field_functions_2,
                                   analytical_solution_2);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_time_integrator(bool const do_restart)
{
  // Setup time integrator

  // For the two-domain solver the parameter start_with_low_order has to be true.
  // This is due to the fact that the setup function of the time integrator initializes
  // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
  // start_with_low_order == false. However, the combined time step size
  // is not known at this point since the two domains have to first communicate with each other
  // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
  // inconsistencies is to preclude the case start_with_low_order == false.
  AssertThrow(param_1.start_with_low_order == true && param_2.start_with_low_order == true,
              ExcMessage("start_with_low_order has to be true for two-domain solver."));

  time_integrator_1->setup(do_restart);
  time_integrator_2->setup(do_restart);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::set_start_time()
{
  // Setup time integrator and get time step size
  double time_1 = param_1.start_time, time_2 = param_2.start_time;

  double time = std::min(time_1, time_2);

  // Set the same time step size for both time integrators (for the two domains)
  time_integrator_1->reset_time(time);
  time_integrator_2->reset_time(time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::synchronize_time_step_size()
{
  // Setup time integrator and get time step size
  double time_step_size_1 = 1.0, time_step_size_2 = 1.0;

  // get time step sizes
  time_step_size_1 = time_integrator_1->get_time_step_size();
  time_step_size_2 = time_integrator_2->get_time_step_size();

  // take the minimum
  double time_step_size = std::min(time_step_size_1, time_step_size_2);

  // decrease time_step in order to exactly hit end_time
  if(use_adaptive_time_stepping == false)
  {
    // assume that domain 1 is the first to start and the last to end
    time_step_size =
      adjust_time_step_to_hit_end_time(param_1.start_time, param_1.end_time, time_step_size);

    pcout << std::endl
          << "Combined time step size for both domains: " << time_step_size << std::endl;
  }

  // set the time step size
  time_integrator_1->set_time_step_size(time_step_size);
  time_integrator_2->set_time_step_size(time_step_size);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::setup_solvers()
{
  // DOMAIN 1
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_1->setup_solvers(
      time_integrator_1->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_1->setup_solvers(
      time_integrator_1->get_time_step_size(),
      time_integrator_1->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_1->setup_solvers(
      time_integrator_1->get_time_step_size(),
      time_integrator_1->get_scaling_factor_time_derivative_term());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // DOMAIN 2
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_2->setup_solvers(
      time_integrator_2->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_2->setup_solvers(
      time_integrator_2->get_time_step_size(),
      time_integrator_2->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_2->setup_solvers(
      time_integrator_2->get_time_step_size(),
      time_integrator_2->get_scaling_factor_time_derivative_term());
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
  bool finished_1 = false, finished_2 = false;

  set_start_time();

  synchronize_time_step_size();

  while(!finished_1 || !finished_2)
  {
    // advance one time step for DOMAIN 1
    finished_1 = time_integrator_1->advance_one_timestep(!finished_1);

    // Note that the coupling of both solvers via the inflow boundary conditions is performed in the
    // postprocessing step of the solver for DOMAIN 1, overwriting the data global structures which
    // are subsequently used by the solver for DOMAIN 2 to evaluate the boundary conditions.

    // advance one time step for DOMAIN 2
    finished_2 = time_integrator_2->advance_one_timestep(!finished_2);

    if(use_adaptive_time_stepping == true)
    {
      // Both domains have already calculated the new, adaptive time step size individually in
      // function advance_one_timestep(). Here, we only have to synchronize the time step size for
      // both domains.
      synchronize_time_step_size();
    }
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
NavierStokesProblem<dim, degree_u, degree_p, Number>::solve_problem(bool const do_restart)
{
  // this function has to be defined in the header file that implements all problem specific things
  // like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions_1(triangulation_1,
                                            n_refine_space_domain1,
                                            boundary_descriptor_velocity_1,
                                            boundary_descriptor_pressure_1,
                                            periodic_faces_1);

  create_grid_and_set_boundary_conditions_2(triangulation_2,
                                            n_refine_space_domain2,
                                            boundary_descriptor_velocity_2,
                                            boundary_descriptor_pressure_2,
                                            periodic_faces_2);

  print_grid_data(
    pcout, n_refine_space_domain1, triangulation_1, n_refine_space_domain2, triangulation_2);

  setup_navier_stokes_operation();

  // setup time integrator before calling setup_solvers (this is necessary since the setup of the
  // solvers depends on quantities such as the time_step_size or gamma0!!!)
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
        // this does in principle work although it doesn't make much sense
        if(REFINE_STEPS_TIME_MIN != REFINE_STEPS_TIME_MAX &&
           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Warning: you are starting from a restart and refine the time steps!"
                    << std::endl;
      }
    }

    // time refinements in order to perform temporal convergence tests
    for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;
        refine_steps_time <= REFINE_STEPS_TIME_MAX;
        ++refine_steps_time)
    {
      NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, VALUE_TYPE>
        navier_stokes_problem(REFINE_STEPS_SPACE_DOMAIN1,
                              REFINE_STEPS_SPACE_DOMAIN2,
                              refine_steps_time);

      navier_stokes_problem.solve_problem(do_restart);
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
