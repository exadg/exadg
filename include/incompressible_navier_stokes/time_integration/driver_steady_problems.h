/*
 * DriverSteadyProblems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

#include <deal.II/base/timer.h>

template<int dim> class PostProcessorBase;

template<int dim, typename value_type, typename NavierStokesOperation>
class DriverSteadyProblems
{
public:
  DriverSteadyProblems(std_cxx11::shared_ptr<NavierStokesOperation>   navier_stokes_operation_in,
                       std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor_in,
                       InputParametersNavierStokes<dim> const         &param_in)
    :
    navier_stokes_operation(navier_stokes_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    total_time(0.0)
  {}

  void setup();

  void solve_steady_problem();

  void analyze_computing_times() const;

private:
  void initialize_vectors();
  void initialize_solution();

  void solve();
  void postprocessing();

  std_cxx11::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor;
  InputParametersNavierStokes<dim> const &param;

  Timer global_timer;
  value_type total_time;

  parallel::distributed::BlockVector<value_type> solution;
  parallel::distributed::BlockVector<value_type> rhs_vector;
  parallel::distributed::Vector<value_type> vorticity;
  parallel::distributed::Vector<value_type> divergence;
};

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using the analytical solution
  // or a guess of the velocity and pressure field
  initialize_solution();
}

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
initialize_vectors()
{
  // solution
  navier_stokes_operation->initialize_block_vector_velocity_pressure(solution);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes)
    navier_stokes_operation->initialize_block_vector_velocity_pressure(rhs_vector);

  // vorticity
  navier_stokes_operation->initialize_vector_vorticity(vorticity);

  // divergence
  if(this->param.output_data.compute_divergence == true)
  {
    navier_stokes_operation->initialize_vector_velocity(divergence);
  }
}

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
initialize_solution()
{
  double time = 0.0;
  navier_stokes_operation->prescribe_initial_conditions(solution.block(0),solution.block(1),time);
}


template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
solve()
{
  Timer timer;
  timer.restart();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Solving steady state problem ..." << std::endl;

  // Steady Stokes equations
  if(this->param.equation_type == EquationType::Stokes)
  {
    // calculate rhs vector
    navier_stokes_operation->rhs_stokes_problem(rhs_vector);

    // solve coupled system of equations
    unsigned int iterations = navier_stokes_operation->solve_linear_stokes_problem(solution,rhs_vector);
    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std:: endl
                << "Solve linear Stokes problem:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    }
  }
  else // Steady Navier-Stokes equations
  {
    // Newton solver
    unsigned int newton_iterations;
    double average_linear_iterations;
    navier_stokes_operation->solve_nonlinear_steady_problem(solution,newton_iterations,average_linear_iterations);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std:: endl
                << "Solve nonlinear Navier-Stokes problem:" << std::endl
                << "  Linear iterations (avg):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << average_linear_iterations << std::endl
                << "  Newton iterations:      " << std::setw(12) << std::right << newton_iterations
                << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    }
  }

  // special case: pure Dirichlet BC's
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.error_data.analytical_solution_available == true)
      navier_stokes_operation->shift_pressure(solution.block(1));
    else // analytical_solution_available == false
      navier_stokes_operation->apply_zero_mean(solution.block(1));
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "... done!" << std::endl;
}

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
solve_steady_problem()
{
  global_timer.restart();

  postprocessing();

  solve();

  postprocessing();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
postprocessing()
{
  // calculate divergence
  if(this->param.output_data.compute_divergence == true)
  {
    navier_stokes_operation->compute_divergence(divergence, solution.block(0));
  }

  // calculate vorticity
  navier_stokes_operation->compute_vorticity(vorticity,solution.block(0));

  this->postprocessor->do_postprocessing(solution.block(0),
                                         solution.block(0), // intermediate_velocity = velocity
                                         solution.block(1),
                                         vorticity,
                                         divergence);
}

template<int dim, typename value_type, typename NavierStokesOperation>
void DriverSteadyProblems<dim, value_type, NavierStokesOperation>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl
        << "_________________________________________________________________________________" << std::endl << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max " << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (this->total_time, MPI_COMM_WORLD);
  pcout << "  Global time:         " << std::scientific
        << std::setprecision(4) << std::setw(10) << data.min << " "
        << std::setprecision(4) << std::setw(10) << data.avg << " "
        << std::setprecision(4) << std::setw(10) << data.max << " "
        << "          " << "  "
        << std::setw(6) << std::left << data.min_index << " "
        << std::setw(6) << std::left << data.max_index << std::endl
        << "_________________________________________________________________________________" << std::endl << std::endl;
}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
