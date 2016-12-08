/*
 * DriverSteadyConvDiff.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DRIVERSTEADYCONVDIFF_H_
#define INCLUDE_DRIVERSTEADYCONVDIFF_H_

#include <deal.II/base/timer.h>

namespace ConvDiff
{
  template<int dim, int fe_degree> class PostProcessor;
  class InputParametersConvDiff;
}

template<int dim, int fe_degree, typename value_type, typename ConvDiffOperation>
class DriverSteadyConvDiff
{
public:
  DriverSteadyConvDiff(std_cxx11::shared_ptr<ConvDiffOperation>                       conv_diff_operation_in,
                       std_cxx11::shared_ptr<ConvDiff::PostProcessor<dim,fe_degree> > postprocessor_in,
                       ConvDiff::InputParametersConvDiff const                        &param_in)
    :
    conv_diff_operation(conv_diff_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    total_time(0.0)
  {}

  void setup()
  {
    // initialize global solution vectors (allocation)
    initialize_vectors();

    // initialize solution by interpolation of initial data
    initialize_solution();
  }

  void solve_problem()
  {
    global_timer.restart();

    postprocessing();

    solve();

    postprocessing();

    total_time += global_timer.wall_time();

    analyze_computing_times();
  }

  void analyze_computing_times() const
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

private:
  void initialize_vectors()
  {
    // solution
    conv_diff_operation->initialize_dof_vector(solution);

    // rhs_vector
    conv_diff_operation->initialize_dof_vector(rhs_vector);
  }

  void initialize_solution()
  {
    double time = 0.0;
    conv_diff_operation->prescribe_initial_conditions(solution,time);
  }


  void solve()
  {
    Timer timer;
    timer.restart();

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << std::endl << "Solving steady state problem ..." << std::endl;

    // calculate rhs vector
    conv_diff_operation->rhs(rhs_vector);

    // solve linear system of equations
    unsigned int iterations = conv_diff_operation->solve(solution,rhs_vector);
    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std:: endl
                << "Solve linear system of equations:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    }

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << std::endl << "... done!" << std::endl;
  }


  void postprocessing()
  {
    this->postprocessor->do_postprocessing(solution);
  }

  std_cxx11::shared_ptr<ConvDiffOperation> conv_diff_operation;
  std_cxx11::shared_ptr<ConvDiff::PostProcessor<dim,fe_degree> > postprocessor;
  ConvDiff::InputParametersConvDiff const &param;

  // timer
  Timer global_timer;
  value_type total_time;

  // vectors
  parallel::distributed::Vector<value_type> solution;
  parallel::distributed::Vector<value_type> rhs_vector;
};

#endif /* INCLUDE_DRIVERSTEADYCONVDIFF_H_ */
