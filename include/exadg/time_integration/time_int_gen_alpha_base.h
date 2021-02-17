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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/time_int_base.h>

namespace ExaDG
{
using namespace dealii;

template<typename Number>
class TimeIntGenAlphaBase : public TimeIntBase
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  TimeIntGenAlphaBase(double const &       start_time_,
                      double const &       end_time_,
                      unsigned int const   max_number_of_time_steps_,
                      double const         spectral_radius_,
                      GenAlphaType const & gen_alpha_type_,
                      RestartData const &  restart_data_,
                      MPI_Comm const &     mpi_comm_,
                      bool const           print_wall_times_);

  double
  get_time_step_size() const override;

  void
  set_current_time_step_size(double const & time_step_size) override;

protected:
  double
  get_scaling_factor_mass() const;

  double
  get_mid_time() const;

  void
  compute_const_vector(VectorType &       const_vector,
                       VectorType const & displacement_n,
                       VectorType const & velocity_n,
                       VectorType const & acceleration_n) const;

  void
  update_displacement(VectorType & displacement_np, VectorType const & displacement_n) const;

  void
  update_velocity(VectorType &       velocity_np,
                  VectorType const & displacement_np,
                  VectorType const & displacement_n,
                  VectorType const & velocity_n,
                  VectorType const & acceleration_n) const;

  void
  update_acceleration(VectorType &       acceleration_np,
                      VectorType const & displacement_np,
                      VectorType const & displacement_n,
                      VectorType const & velocity_n,
                      VectorType const & acceleration_n) const;

private:
  void
  do_timestep_pre_solve(bool const print_header) override;

  void
  do_timestep_post_solve() override;

  virtual void
  prepare_vectors_for_next_timestep() = 0;

  /*
   * returns whether solver info has to be written in the current time step.
   */
  virtual bool
  print_solver_info() const = 0;

  double spectral_radius;
  double alpha_m;
  double alpha_f;
  double beta;
  double gamma;

  double time_step;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_ */
