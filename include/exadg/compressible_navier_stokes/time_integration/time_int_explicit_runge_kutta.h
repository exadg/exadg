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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/explicit_runge_kutta.h>
#include <exadg/time_integration/ssp_runge_kutta.h>
#include <exadg/time_integration/time_int_explicit_runge_kutta_base.h>

namespace ExaDG
{
namespace CompNS
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
class TimeIntExplRK : public TimeIntExplRKBase<Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntExplRK(std::shared_ptr<Operator>                       operator_in,
                Parameters const &                              param_in,
                MPI_Comm const &                                mpi_comm_in,
                bool const                                      print_wall_times_in,
                std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  initialize_time_integrator();

  void
  initialize_vectors();

  void
  initialize_solution();

  void
  detect_instabilities() const;

  void
  postprocessing() const;

  void
  do_timestep_solve() final;

  bool
  print_solver_info() const;

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  calculate_pressure();

  void
  calculate_velocity();

  void
  calculate_temperature();

  void
  calculate_vorticity();

  void
  calculate_divergence();

  std::shared_ptr<Operator> pde_operator;

  std::shared_ptr<ExplicitTimeIntegrator<Operator, VectorType>> rk_time_integrator;

  Parameters const & param;

  unsigned int const refine_steps_time;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // monitor the L2-norm of the solution vector in order to detect instabilities
  mutable double l2_norm;

  // time step calculation
  double const cfl_number;
  double const diffusion_number;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ \
        */
