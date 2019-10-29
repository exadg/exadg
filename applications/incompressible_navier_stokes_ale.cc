/*
 * incompressible_navier_stokes.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>


// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor_base.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/interface.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/driver_steady_problems.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

// ALE
#include "../include/incompressible_navier_stokes/spatial_discretization/moving_mesh.h"

using namespace dealii;
using namespace IncNS;

// specify the flow problem that has to be solved


// 2D Navier-Stokes flow
//#include "incompressible_navier_stokes_ale_test_cases/poiseuille_ale.h"
#include "incompressible_navier_stokes_ale_test_cases/vortex_ale.h"
//#include "incompressible_navier_stokes_ale_test_cases/taylor_vortex_ale.h"
//#include "incompressible_navier_stokes_ale_test_cases/free_stream_preservation_test.h"
//#include "incompressible_navier_stokes_ale_test_cases/turbulent_channel_ale.h"
//#include "incompressible_navier_stokes_ale_test_cases/3D_taylor_green_vortex_ale.h"


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
  solve() = 0;

  virtual void
  analyze_computing_times() const = 0;
};

template<int dim, typename Number>
class Problem : public ProblemBase<Number>
{
public:
  Problem();

  void
  setup(InputParameters const & param);

  void
  solve();

  void
  analyze_computing_times() const;

private:
  void
  print_header() const;

  ConditionalOStream pcout;

  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<FieldFunctions<dim>>        field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>>   boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>>   boundary_descriptor_pressure;
  std::shared_ptr<MeshMovementFunctions<dim>> mesh_movement_function;

  InputParameters param;

  typedef DGNavierStokesBase<dim, Number>               DGBase;
  typedef DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation;

  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  // unsteady solvers
  typedef TimeIntBDF<Number>                   TimeInt;
  typedef TimeIntBDFCoupled<Number>            TimeIntCoupled;
  typedef TimeIntBDFDualSplitting<Number>      TimeIntDualSplitting;
  typedef TimeIntBDFPressureCorrection<Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> time_integrator;

  // steady solver
  typedef DriverSteadyProblems<Number> DriverSteady;

  std::shared_ptr<DriverSteady> driver_steady;

  typedef MovingMesh<dim, Number> DGALE;
  std::shared_ptr<DGALE>          ale_operation;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem()
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
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
  << "                     incompressible Navier-Stokes equations                      " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Problem<dim, Number>::setup(InputParameters const & param_in)
{
  timer.restart();

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

  create_grid_and_set_boundary_ids(triangulation, param.h_refinements, periodic_faces);
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor_velocity.reset(new BoundaryDescriptorU<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions(boundary_descriptor_velocity, boundary_descriptor_pressure);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  set_field_functions(field_functions);

  // set mesh movement function
  if(param.ale_formulation == true)
    mesh_movement_function = set_mesh_movement_function<dim>();

  // initialize postprocessor
  postprocessor = construct_postprocessor<dim, Number>(param);

  if(param.solver_type == SolverType::Unsteady)
  {
    // initialize navier_stokes_operation
    if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

      navier_stokes_operation_coupled.reset(new DGCoupled(*triangulation, param, postprocessor));

      navier_stokes_operation = navier_stokes_operation_coupled;

      time_integrator.reset(new TimeIntCoupled(navier_stokes_operation_coupled,
                                               navier_stokes_operation_coupled,
                                               param));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting;

      navier_stokes_operation_dual_splitting.reset(
        new DGDualSplitting(*triangulation, param, postprocessor));

      navier_stokes_operation = navier_stokes_operation_dual_splitting;

      time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting,
                                                     navier_stokes_operation_dual_splitting,
                                                     param));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction;

      navier_stokes_operation_pressure_correction.reset(
        new DGPressureCorrection(*triangulation, param, postprocessor));

      navier_stokes_operation = navier_stokes_operation_pressure_correction;

      time_integrator.reset(
        new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction,
                                      navier_stokes_operation_pressure_correction,
                                      param));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
  else if(param.solver_type == SolverType::Steady)
  {
    // initialize navier_stokes_operation
    std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

    navier_stokes_operation_coupled.reset(new DGCoupled(*triangulation, param, postprocessor));

    navier_stokes_operation = navier_stokes_operation_coupled;

    // initialize driver for steady state problem that depends on navier_stokes_operation
    driver_steady.reset(
      new DriverSteady(navier_stokes_operation_coupled, navier_stokes_operation_coupled, param));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(param.ale_formulation == true)
    ale_operation = std::make_shared<DGALE>(param,
                                            triangulation,
                                            mesh_movement_function,
                                            navier_stokes_operation);


  // Depends on mapping which is initialized in constructor of MovingMesh
  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));
  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  // depends on matrix free which is initialized in navier_stokes_operation->setup
  if(param.ale_formulation == true)
  {
    AssertThrow(ale_operation.get() != 0, ExcMessage("Not initialized."));
    ale_operation->setup();
    if(param.calculation_of_time_step_size == TimeStepCalculation::CFL &&
       param.adaptive_time_stepping == true)
      time_integrator->set_grid_velocity_cfl(ale_operation->get_grid_velocity());
  }

  if(param.solver_type == SolverType::Unsteady)
  {
    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!!!)
    time_integrator->setup(param.restarted_simulation);

    navier_stokes_operation->setup_solvers(
      time_integrator->get_scaling_factor_time_derivative_term(), time_integrator->get_velocity());
  }
  else if(param.solver_type == SolverType::Steady)
  {
    driver_steady->setup();

    navier_stokes_operation->setup_solvers(1.0 /* dummy */, driver_steady->get_velocity());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(param.ale_formulation == true && param.start_with_low_order == false &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    std::vector<double> eval_times(param.order_time_integrator);

    for(unsigned int i = 0; i < param.order_time_integrator; ++i)
      eval_times[i] = time_integrator->get_previous_time(i);

    if(param.grid_velocity_analytical == false)
      ale_operation->initialize_grid_coordinates_on_former_mesh_instances(eval_times);

    time_integrator->set_former_solution_considering_former_mesh_instances(
      ale_operation->get_former_solution_on_former_mesh_instances(eval_times));

    if(param.convective_problem())
      time_integrator->set_convective_term_considering_former_mesh_instances(
        ale_operation->get_convective_term_on_former_mesh_instances(eval_times));

    // Dual splitting
    if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      auto time_integrator_ds = std::dynamic_pointer_cast<TimeIntDualSplitting>(time_integrator);

      if(param.convective_problem())
      {
        if(param.divu_integrated_by_parts == true && param.divu_use_boundary_data == true)
        {
          time_integrator_ds
            ->set_vec_rhs_ppe_div_term_convective_term_considering_former_mesh_instances(
              ale_operation->get_vec_rhs_ppe_div_term_convective_term_on_former_mesh_instances(
                eval_times));
        }

        time_integrator_ds->set_vec_rhs_ppe_convective_considering_former_mesh_instances(
          ale_operation->get_vec_rhs_ppe_convective_on_former_mesh_instances(eval_times));
      }

      if(param.viscous_problem())
      {
        time_integrator_ds->set_vec_rhs_ppe_viscous_considering_former_mesh_instances(
          ale_operation->get_vec_rhs_ppe_viscous_on_former_mesh_instances(eval_times));
      }
    }

    // Pressure-correction
    if(param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      auto time_integrator_pc =
        std::dynamic_pointer_cast<TimeIntPressureCorrection>(time_integrator);

      if(param.order_pressure_extrapolation > 0)
      {
        time_integrator_pc->set_vec_pressure_gradient_term_considering_former_mesh_instances(
          ale_operation->get_vec_pressure_gradient_term_on_former_mesh_instances(eval_times));

        time_integrator_pc->set_pressure_mass_matrix_considering_former_mesh_instances(
          ale_operation->get_pressure_mass_matrix_term_on_former_mesh_instances(eval_times));
      }
    }
  }

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve()
{
  if(param.solver_type == SolverType::Unsteady)
  {
    // stability analysis (uncomment if desired)
    // time_integrator->postprocessing_stability_analysis();

    // run time loop
    if(this->param.problem_type == ProblemType::Steady)
      time_integrator->timeloop_steady_problem();
    else if(this->param.problem_type == ProblemType::Unsteady && param.ale_formulation == false)
      time_integrator->timeloop();
    else if(this->param.problem_type == ProblemType::Unsteady && param.ale_formulation == true)
    {
      bool timeloop_finished = false;

      while(!timeloop_finished)
      {
        // BDF coefficients are updated within advance_one_timestep()
        // for ALE it is already needed before the function call to compute the grid velocity
        time_integrator->update_time_integrator_constants();

        // calculate mesh at time t_n+1
        ale_operation->move_mesh(time_integrator->get_next_time());

        // calculate grid velocity at time t_n+1 and hand it over to navier_stokes_operation
        // for correct evaluation of convective term
        ale_operation->update_grid_velocities(
          time_integrator->get_next_time(),
          time_integrator->get_time_step_size(),
          time_integrator->get_current_time_integrator_constants());

        // set grid velocity at time t_n+1 since the time integrator needs the grid velocity
        // to compute the time step size for the next time step at the end of the current time step
        if(param.calculation_of_time_step_size == TimeStepCalculation::CFL &&
           param.adaptive_time_stepping == true)
          time_integrator->set_grid_velocity_cfl(ale_operation->get_grid_velocity());

        // advance from t_n -> t_n+1
        timeloop_finished = time_integrator->advance_one_timestep(!timeloop_finished);
      }
    }
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }
  else if(param.solver_type == SolverType::Steady)
  {
    // solve steady problem
    driver_steady->solve_steady_problem();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for incompressible Navier-Stokes solver:" << std::endl;

  // Iterations
  if(param.solver_type == SolverType::Unsteady)
  {
    this->pcout << std::endl << "Average number of iterations:" << std::endl;

    std::vector<std::string> names;
    std::vector<double>      iterations;

    this->time_integrator->get_iterations(names, iterations);

    unsigned int length = 1;
    for(unsigned int i = 0; i < names.size(); ++i)
    {
      length = length > names[i].length() ? length : names[i].length();
    }

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(param.solver_type == SolverType::Unsteady)
  {
    this->time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    this->driver_steady->get_wall_times(names, computing_times);
  }

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(25) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data =
    Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD);
  double const setup_time_avg = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(25) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double compute_and_set_mesh_velocity = 0.0;
  double update                        = 0.0;
  double advance_mesh                  = 0.0;
  if(param.ale_formulation == true)
  {
    Utilities::MPI::MinMaxAvg advance_mesh_time_data =
      Utilities::MPI::min_max_avg(ale_operation->get_wall_time_advance_mesh(), MPI_COMM_WORLD);
    advance_mesh = advance_mesh_time_data.avg;
    this->pcout << "  " << std::setw(length + 2) << std::left << "Move Mesh" << std::setprecision(2)
                << std::scientific << std::setw(25) << std::right << advance_mesh << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << advance_mesh / overall_time_avg * 100 << " %" << std::endl;

    Utilities::MPI::MinMaxAvg update_time_data =
      Utilities::MPI::min_max_avg(ale_operation->get_wall_time_ale_update(), MPI_COMM_WORLD);
    update = update_time_data.avg;
    this->pcout << "  " << std::setw(length + 2) << std::left << "Update" << std::setprecision(2)
                << std::scientific << std::setw(25) << std::right << update << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << update / overall_time_avg * 100 << " %" << std::endl;

    Utilities::MPI::MinMaxAvg compute_and_set_mesh_velocity_time_data =
      Utilities::MPI::min_max_avg(ale_operation->get_wall_time_compute_and_set_mesh_velocity(),
                                  MPI_COMM_WORLD);
    compute_and_set_mesh_velocity = compute_and_set_mesh_velocity_time_data.avg;
    this->pcout << "  " << std::setw(length + 2) << std::left << "Compute and set mesh velocity"
                << std::setprecision(2) << std::scientific << std::setw(12) << std::right
                << compute_and_set_mesh_velocity << " s  " << std::setprecision(2) << std::fixed
                << std::setw(6) << std::right
                << compute_and_set_mesh_velocity / overall_time_avg * 100 << " %" << std::endl;
  }

  double const other = overall_time_avg - sum_of_substeps - setup_time_avg - update - advance_mesh -
                       compute_and_set_mesh_velocity;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(25) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;


  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(25) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  this->pcout << std::endl
              << "Computational costs (including setup + postprocessing):" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs = this->navier_stokes_operation->get_number_of_dofs();

  if(param.solver_type == SolverType::Unsteady)
  {
    unsigned int N_time_steps      = this->time_integrator->get_number_of_time_steps();
    double const time_per_timestep = overall_time_avg / (double)N_time_steps;
    this->pcout << std::endl
                << "Throughput per time step (including setup + postprocessing):" << std::endl
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
    this->pcout << std::endl
                << "Throughput (including setup + postprocessing):" << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (overall_time_avg * N_mpi_processes) << " DoFs/s/core" << std::endl;
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

    // set parameters
    InputParameters param;
    set_input_parameters(param);

    // check parameters in case of restart
    if(param.restarted_simulation)
    {
      AssertThrow(DEGREE_MIN == DEGREE_MAX && REFINE_SPACE_MIN == REFINE_SPACE_MAX,
                  ExcMessage("Spatial refinement not possible in combination with restart!"));

      AssertThrow(REFINE_TIME_MIN == REFINE_TIME_MAX,
                  ExcMessage("Temporal refinement not possible in combination with restart!"));
    }

    // k-refinement
    for(unsigned int degree = DEGREE_MIN; degree <= DEGREE_MAX; ++degree)
    {
      // h-refinement
      for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
          ++h_refinements)
      {
        // dt-refinement
        for(unsigned int dt_refinements = REFINE_TIME_MIN; dt_refinements <= REFINE_TIME_MAX;
            ++dt_refinements)
        {
          // reset degree
          param.degree_u = degree;

          // reset mesh refinements
          param.h_refinements = h_refinements;

          // reset dt refinements
          param.dt_refinements = dt_refinements;

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

          problem->solve();

          problem->analyze_computing_times();
        }
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
