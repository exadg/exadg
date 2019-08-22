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
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor_base.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/interface.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem that has to be solved

// template
#include "incompressible_navier_stokes_test_cases/template_two_domains.h"

//#include "incompressible_navier_stokes_test_cases/turbulent_channel_two_domains.h"
//#include "incompressible_navier_stokes_test_cases/backward_facing_step_two_domains.h"
//#include "incompressible_navier_stokes_test_cases/fda_nozzle_benchmark.h"

template<typename Number>
class ProblemBase
{
public:
  virtual ~ProblemBase()
  {
  }

  virtual void
  setup(InputParameters const & param_1_in, InputParameters const & param_2_in) = 0;

  virtual void
  solve() const = 0;

  virtual void
  analyze_computing_times() const = 0;
};

template<int dim, typename Number>
class Problem : public ProblemBase<Number>
{
public:
  Problem();

  void
  setup(InputParameters const & param_1_in, InputParameters const & param_2_in);

  void
  solve() const;

  void
  analyze_computing_times() const;

private:
  void
  print_header() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  ConditionalOStream pcout;

  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation_1, triangulation_2;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces_1, periodic_faces_2;

  bool use_adaptive_time_stepping;

  std::shared_ptr<FieldFunctions<dim>>      field_functions_1, field_functions_2;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity_1,
    boundary_descriptor_velocity_2;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure_1,
    boundary_descriptor_pressure_2;

  InputParameters param_1, param_2;

  typedef DGNavierStokesBase<dim, Number>               DGBase;
  typedef DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation_1, navier_stokes_operation_2;

  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor_1, postprocessor_2;

  typedef TimeIntBDF<Number>                   TimeInt;
  typedef TimeIntBDFCoupled<Number>            TimeIntCoupled;
  typedef TimeIntBDFDualSplitting<Number>      TimeIntDualSplitting;
  typedef TimeIntBDFPressureCorrection<Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> time_integrator_1, time_integrator_2;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  unsigned int const length = 15;

  void
  analyze_iterations(InputParameters const &        param,
                     std::shared_ptr<TimeInt> const time_integrator) const;

  double
  analyze_computing_times(InputParameters const &        param,
                          std::shared_ptr<TimeInt> const time_integrator) const;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem()
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    use_adaptive_time_stepping(false),
    overall_time(0.0),
    setup_time(0.0)
{
}

template<int dim, typename Number>
void
Problem<dim, Number>::print_header() const
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

template<int dim, typename Number>
void
Problem<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double time_1 = param_1.start_time, time_2 = param_2.start_time;

  double time = std::min(time_1, time_2);

  // Set the same time step size for both time integrators (for the two domains)
  time_integrator_1->reset_time(time);
  time_integrator_2->reset_time(time);
}

template<int dim, typename Number>
void
Problem<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_1 = std::numeric_limits<double>::max();
  double time_step_size_2 = std::numeric_limits<double>::max();

  // get time step sizes
  if(use_adaptive_time_stepping == true)
  {
    if(time_integrator_1->get_time() > param_1.start_time - EPSILON)
      time_step_size_1 = time_integrator_1->get_time_step_size();

    if(time_integrator_2->get_time() > param_2.start_time - EPSILON)
      time_step_size_2 = time_integrator_2->get_time_step_size();
  }
  else
  {
    time_step_size_1 = time_integrator_1->get_time_step_size();
    time_step_size_2 = time_integrator_2->get_time_step_size();
  }

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

template<int dim, typename Number>
void
Problem<dim, Number>::setup(InputParameters const & param_1_in, InputParameters const & param_2_in)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout);

  param_1 = param_1_in;
  param_1.check_input_parameters();
  param_1.print(pcout, "List of input parameters for DOMAIN 1:");

  param_2 = param_2_in;
  param_2.check_input_parameters();
  param_2.print(pcout, "List of input parameters for DOMAIN 2:");

  AssertThrow(param_1.dim == param_2.dim, ExcMessage("Invalid parameters."));

  // triangulation
  if(param_1.triangulation_type == TriangulationType::Distributed)
  {
    triangulation_1.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param_1.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation_1.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  if(param_2.triangulation_type == TriangulationType::Distributed)
  {
    triangulation_2.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param_2.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation_2.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  // create grid
  create_grid_and_set_boundary_ids_1(triangulation_1, param_1.h_refinements, periodic_faces_1);
  create_grid_and_set_boundary_ids_2(triangulation_2, param_2.h_refinements, periodic_faces_2);

  print_grid_data(
    pcout, param_1.h_refinements, *triangulation_1, param_2.h_refinements, *triangulation_2);

  boundary_descriptor_velocity_1.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_1.reset(new BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions_1(boundary_descriptor_velocity_1, boundary_descriptor_pressure_1);

  boundary_descriptor_velocity_2.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure_2.reset(new BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions_2(boundary_descriptor_velocity_2, boundary_descriptor_pressure_2);


  field_functions_1.reset(new FieldFunctions<dim>());
  field_functions_2.reset(new FieldFunctions<dim>());
  set_field_functions_1(field_functions_1);
  set_field_functions_2(field_functions_2);

  // constant vs. adaptive time stepping
  use_adaptive_time_stepping = param_1.adaptive_time_stepping;

  AssertThrow(param_1.calculation_of_time_step_size == param_2.calculation_of_time_step_size,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(param_1.adaptive_time_stepping == param_2.adaptive_time_stepping,
              ExcMessage("Type of time step calculation has to be the same for both domains."));

  AssertThrow(param_1.solver_type == SolverType::Unsteady &&
                param_2.solver_type == SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor_1 = construct_postprocessor<dim, Number>(param_1, 1);
  postprocessor_2 = construct_postprocessor<dim, Number>(param_2, 2);

  // initialize navier_stokes_operation_1 (DOMAIN 1)
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    std::shared_ptr<DGCoupled> navier_stokes_operation_coupled_1;

    navier_stokes_operation_coupled_1.reset(
      new DGCoupled(*triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_coupled_1;

    time_integrator_1.reset(new TimeIntCoupled(navier_stokes_operation_coupled_1,
                                               navier_stokes_operation_coupled_1,
                                               param_1));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting_1;

    navier_stokes_operation_dual_splitting_1.reset(
      new DGDualSplitting(*triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_dual_splitting_1;

    time_integrator_1.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting_1,
                                                     navier_stokes_operation_dual_splitting_1,
                                                     param_1));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction_1;

    navier_stokes_operation_pressure_correction_1.reset(
      new DGPressureCorrection(*triangulation_1, param_1, postprocessor_1));

    navier_stokes_operation_1 = navier_stokes_operation_pressure_correction_1;

    time_integrator_1.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction_1,
                                    navier_stokes_operation_pressure_correction_1,
                                    param_1));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize navier_stokes_operation_2 (DOMAIN 2)
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    std::shared_ptr<DGCoupled> navier_stokes_operation_coupled_2;

    navier_stokes_operation_coupled_2.reset(
      new DGCoupled(*triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_coupled_2;

    time_integrator_2.reset(new TimeIntCoupled(navier_stokes_operation_coupled_2,
                                               navier_stokes_operation_coupled_2,
                                               param_2));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting_2;

    navier_stokes_operation_dual_splitting_2.reset(
      new DGDualSplitting(*triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_dual_splitting_2;

    time_integrator_2.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting_2,
                                                     navier_stokes_operation_dual_splitting_2,
                                                     param_2));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction_2;

    navier_stokes_operation_pressure_correction_2.reset(
      new DGPressureCorrection(*triangulation_2, param_2, postprocessor_2));

    navier_stokes_operation_2 = navier_stokes_operation_pressure_correction_2;

    time_integrator_2.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction_2,
                                    navier_stokes_operation_pressure_correction_2,
                                    param_2));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // setup navier_stokes_operation

  AssertThrow(navier_stokes_operation_1.get() != 0, ExcMessage("Not initialized."));
  AssertThrow(navier_stokes_operation_2.get() != 0, ExcMessage("Not initialized."));

  navier_stokes_operation_1->setup(periodic_faces_1,
                                   boundary_descriptor_velocity_1,
                                   boundary_descriptor_pressure_1,
                                   field_functions_1);

  navier_stokes_operation_2->setup(periodic_faces_2,
                                   boundary_descriptor_velocity_2,
                                   boundary_descriptor_pressure_2,
                                   field_functions_2);

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

  // setup time integrator before calling setup_solvers (this is necessary since the setup of the
  // solvers depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator_1->setup(param_1.restarted_simulation);
  time_integrator_2->setup(param_2.restarted_simulation);

  // setup solvers

  navier_stokes_operation_1->setup_solvers(
    time_integrator_1->get_scaling_factor_time_derivative_term(),
    time_integrator_1->get_velocity());

  navier_stokes_operation_2->setup_solvers(
    time_integrator_2->get_scaling_factor_time_derivative_term(),
    time_integrator_2->get_velocity());

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve() const
{
  // run time loop

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

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_iterations(InputParameters const &        param,
                                         std::shared_ptr<TimeInt> const time_integrator) const
{
  // Iterations
  if(param.solver_type == SolverType::Unsteady)
  {
    std::vector<std::string> names;
    std::vector<double>      iterations;

    time_integrator->get_iterations(names, iterations);

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }
}
template<int dim, typename Number>
double
Problem<dim, Number>::analyze_computing_times(InputParameters const &        param,
                                              std::shared_ptr<TimeInt> const time_integrator) const
{
  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(param.solver_type == SolverType::Unsteady)
  {
    time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter"));
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  return sum_of_substeps;
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << std::endl
              << "Average number of iterations for incompressible Navier-Stokes solver:"
              << std::endl;

  this->pcout << std::endl << "Domain 1:" << std::endl;

  analyze_iterations(param_1, time_integrator_1);

  this->pcout << std::endl << "Domain 2:" << std::endl;

  analyze_iterations(param_2, time_integrator_2);

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  this->pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  // wall times
  this->pcout << std::endl << "Domain 1:" << std::endl;

  double const time_domain_1_avg = analyze_computing_times(param_1, time_integrator_1);

  // wall times
  this->pcout << std::endl << "Domain 2:" << std::endl;

  double const time_domain_2_avg = analyze_computing_times(param_2, time_integrator_2);

  this->pcout << std::endl;

  Utilities::MPI::MinMaxAvg setup_time_data =
    Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD);
  double const setup_time_avg = setup_time_data.avg;
  this->pcout << "  " << std::setw(length) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - time_domain_1_avg - time_domain_2_avg - setup_time_avg;
  this->pcout << "  " << std::setw(length) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  this->pcout << std::endl
              << "Computational costs (both domains, including setup + postprocessing):"
              << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs = navier_stokes_operation_1->get_number_of_dofs() +
                                       navier_stokes_operation_2->get_number_of_dofs();

  if(param_1.solver_type == SolverType::Unsteady)
  {
    unsigned int N_time_steps      = time_integrator_1->get_number_of_time_steps();
    double const time_per_timestep = overall_time_avg / (double)N_time_steps;
    this->pcout << std::endl
                << "Throughput per time step (both domains, including setup + postprocessing):"
                << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Time steps              = " << std::left << N_time_steps << std::endl
                << "  Wall time per time step = " << std::scientific << std::setprecision(2)
                << time_per_timestep << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter"));
  }

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    InputParameters param_1, param_2;
    set_input_parameters(param_1, 1);
    set_input_parameters(param_2, 2);

    AssertThrow(param_1.dim == param_2.dim, ExcMessage("Invalid parameters!"));
    AssertThrow(param_1.restarted_simulation == param_2.restarted_simulation,
                ExcMessage("Invalid parameters!"));

    // setup problem and run simulation
    typedef double                       Number;
    std::shared_ptr<ProblemBase<Number>> problem;

    if(param_1.dim == 2)
      problem.reset(new Problem<2, Number>());
    else if(param_1.dim == 3)
      problem.reset(new Problem<3, Number>());
    else
      AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

    problem->setup(param_1, param_2);

    problem->solve();

    problem->analyze_computing_times();
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
