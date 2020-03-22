/*
 * driver_quasi_static_problems.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_
#define INCLUDE_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../postprocessor/postprocessor.h"
#include "../spatial_discretization/operator.h"

using namespace dealii;

namespace Structure
{
class Time
{
public:
  Time(const double end_time, const double delta_t, MPI_Comm comm = MPI_COMM_SELF)
    : time_step_number(0),
      time(0.0),
      end_time(end_time),
      delta_t(delta_t),
      pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  {
  }

  double
  get_current_time() const
  {
    return time;
  }

  double
  get_end_time() const
  {
    return end_time;
  }

  double
  get_delta_t() const
  {
    return delta_t;
  }

  unsigned int
  get_timestep() const
  {
    return time_step_number;
  }

  void
  set_delta_t(double const delta_t_in)
  {
    delta_t = delta_t_in;
  }


  bool
  finished() const
  {
    return this->get_current_time() >= this->get_end_time(); // TODO
  }

  void
  print_header() const
  {
    pcout << std::endl;
    pcout << "--------------------------------------------------------------------------------"
          << std::endl;
    pcout << "  Timestep: " << time_step_number << "  Time: " << time << std::endl;
    pcout << "--------------------------------------------------------------------------------"
          << std::endl;
  }

  void
  do_increment()
  {
    time += delta_t;
    ++time_step_number;
  }

private:
  unsigned int       time_step_number;
  double             time;
  const double       end_time;
  double             delta_t;
  ConditionalOStream pcout;
};

template<int dim, typename Number>
class DriverQuasiStatic
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  DriverQuasiStatic(std::shared_ptr<Operator<dim, Number>>      operator_in,
                    std::shared_ptr<PostProcessor<dim, Number>> postprocessor_in,
                    InputParameters const &                     param_in,
                    MPI_Comm const &                            mpi_comm_in);

  void
  setup();

  void
  solve_problem();

  void
  adapt_time_step_size(unsigned int n_current_iteration,
                       double       current_delta_t,
                       unsigned int ideal_iterations,
                       double       p);

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  unsigned int
  solve_primary();

  void
  postprocessing() const;

  std::shared_ptr<Operator<dim, Number>> pde_operator;

  std::shared_ptr<PostProcessor<dim, Number>> postprocessor;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  std::vector<double> computing_times;

  Time time;

  mutable std::vector<double> norm_vector;

  // vectors
  VectorType solution;
  VectorType rhs_vector;
  VectorType qph_stress;
};

} // namespace Structure

#endif
