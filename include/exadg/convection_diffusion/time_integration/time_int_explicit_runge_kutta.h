/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/explicit_runge_kutta.h>
#include <exadg/time_integration/time_int_explicit_runge_kutta_base.h>

namespace ExaDG
{
namespace ConvDiff
{
// forward declarations
class Parameters;

template<typename Number>
class PostProcessorInterface;

namespace Interface
{
template<typename Number>
class Operator;
}

template<typename Number>
class OperatorExplRK;

template<typename Number>
class TimeIntExplRK : public TimeIntExplRKBase<Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  TimeIntExplRK(std::shared_ptr<Interface::Operator<Number>>    operator_in,
                std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                Parameters const &                              param_in,
                MPI_Comm const &                                mpi_comm_in,
                bool const                                      is_test_in);

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

private:
  void
  initialize_vectors() final;

  void
  initialize_solution() final;

  void
  postprocessing() const final;

  bool
  print_solver_info() const final;

  void
  do_timestep_solve() final;

  void
  calculate_time_step_size() final;

  double
  recalculate_time_step_size() const final;

  void
  initialize_time_integrator() final;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<OperatorExplRK<Number>> expl_rk_operator;

  std::shared_ptr<ExplicitTimeIntegrator<OperatorExplRK<Number>, VectorType>> rk_time_integrator;

  Parameters const & param;

  unsigned int const refine_steps_time;

  std::vector<VectorType const *> velocities;
  std::vector<double>             times;

  // store time step size according to diffusion condition so that it does not have to be
  // recomputed in case of adaptive time stepping
  double time_step_diff;

  double const cfl;
  double const diffusion_number;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
