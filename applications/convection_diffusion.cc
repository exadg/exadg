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

// temporal discretization
#include "convection_diffusion/time_integration/driver_steady_problems.h"
#include "convection_diffusion/time_integration/time_int_bdf.h"
#include "convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor_base.h"

// user interface, etc.
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"


// specify the test case that has to be solved

// template
#include "convection_diffusion_test_cases/template.h"

// convection problems

//#include "convection_diffusion_test_cases/propagating_sine_wave.h"
//#include "convection_diffusion_test_cases/rotating_hill.h"
//#include "convection_diffusion_test_cases/deforming_hill.h"

// diffusion problems

//#include "convection_diffusion_test_cases/diffusive_problem.h"

// convection-diffusion problems

//#include "convection_diffusion_test_cases/constant_rhs.h"
//#include "convection_diffusion_test_cases/boundary_layer_problem.h"
//#include "convection_diffusion_test_cases/const_rhs_const_and_circular_wind.h"

using namespace dealii;
using namespace ConvDiff;

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

template<int dim, typename Number = double>
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

  std::shared_ptr<TimeIntBase> time_integrator;

  std::shared_ptr<DriverSteadyProblems<Number>> driver_steady;

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
  timer.restart();

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

  // initialize convection-diffusion operator
  conv_diff_operator.reset(new DGOperator<dim, Number>(*triangulation, param, postprocessor));
  conv_diff_operator->setup(periodic_faces, boundary_descriptor, field_functions);

  if(param.problem_type == ProblemType::Unsteady)
  {
    // initialize time integrator
    if(param.temporal_discretization == TemporalDiscretization::ExplRK)
    {
      time_integrator.reset(new TimeIntExplRK<Number>(conv_diff_operator, param));
    }
    else if(param.temporal_discretization == TemporalDiscretization::BDF)
    {
      time_integrator.reset(new TimeIntBDF<Number>(conv_diff_operator, param));
    }
    else
    {
      AssertThrow(param.temporal_discretization == TemporalDiscretization::ExplRK ||
                    param.temporal_discretization == TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }

    time_integrator->setup(param.restarted_simulation);
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    // initialize driver for steady convection-diffusion problems
    driver_steady.reset(new DriverSteadyProblems<Number>(conv_diff_operator, param));
    driver_steady->setup();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  VectorType const * velocity_ptr = nullptr;
  VectorType velocity;

  // setup solvers in case of BDF time integration or steady problems
  if(param.problem_type == ProblemType::Unsteady)
  {
    if(param.temporal_discretization == TemporalDiscretization::ExplRK)
    {
      conv_diff_operator->setup_operators_and_solver(
        /* no parameter since it is not used for ExplRK */);
    }
    else if(param.temporal_discretization == TemporalDiscretization::BDF)
    {
      std::shared_ptr<TimeIntBDF<Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<TimeIntBDF<Number>>(time_integrator);

      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        conv_diff_operator->initialize_dof_vector_velocity(velocity);
        conv_diff_operator->interpolate_velocity(velocity, time_integrator->get_time());
        velocity_ptr = &velocity;
      }

      conv_diff_operator->setup_operators_and_solver(
        time_integrator_bdf->get_scaling_factor_time_derivative_term(),
        velocity_ptr);
    }
    else
    {
      AssertThrow(param.temporal_discretization == TemporalDiscretization::ExplRK ||
                    param.temporal_discretization == TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      conv_diff_operator->initialize_dof_vector_velocity(velocity);
      conv_diff_operator->interpolate_velocity(velocity, 0.0 /* time */);
      velocity_ptr = &velocity;
    }

    conv_diff_operator->setup_operators_and_solver(
      1.0 /* scaling_factor_time_derivative_term */,
      velocity_ptr);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve()
{
  if(param.problem_type == ProblemType::Unsteady)
  {
    time_integrator->timeloop();
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->solve_problem();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
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

  this->pcout << "Performance results for convection-diffusion solver:" << std::endl;

  // Iterations are only relevant for BDF time integrator
  if(param.temporal_discretization == TemporalDiscretization::BDF)
  {
    // Iterations
    if(param.problem_type == ProblemType::Unsteady)
    {
      this->pcout << std::endl << "Average number of iterations:" << std::endl;

      std::vector<std::string> names;
      std::vector<double>      iterations;

      std::shared_ptr<TimeIntBDF<Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<TimeIntBDF<Number>>(time_integrator);
      time_integrator_bdf->get_iterations(names, iterations);

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
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(param.problem_type == ProblemType::Unsteady)
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
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data =
    Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD);
  double const setup_time_avg = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - sum_of_substeps - setup_time_avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
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
  types::global_dof_index const DoFs = conv_diff_operator->get_number_of_dofs();

  if(param.problem_type == ProblemType::Unsteady)
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
    ConvDiff::InputParameters param;
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
          param.degree = degree;

          // reset mesh refinement
          param.h_refinements = h_refinements;

          // reset dt_refinements
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
