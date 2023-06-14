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

// ExaDG
#include <exadg/time_integration/time_int_gen_alpha_base.h>

namespace ExaDG
{
template<typename Number>
TimeIntGenAlphaBase<Number>::TimeIntGenAlphaBase(double const &       start_time_,
                                                 double const &       end_time_,
                                                 unsigned int const   max_number_of_time_steps_,
                                                 double const         spectral_radius_,
                                                 GenAlphaType const & gen_alpha_type_,
                                                 RestartData const &  restart_data_,
                                                 MPI_Comm const &     mpi_comm_,
                                                 bool const           is_test_)
  : TimeIntBase(start_time_,
                end_time_,
                max_number_of_time_steps_,
                restart_data_,
                mpi_comm_,
                is_test_),
    spectral_radius(spectral_radius_),
    alpha_m(0.0),
    alpha_f(0.0),
    beta(0.25),
    gamma(0.5),
    time_step(1.0)
{
  switch(gen_alpha_type_)
  {
    case GenAlphaType::Newmark:
      alpha_m = 0.0;
      alpha_f = 0.0;
      beta    = 1.0 / std::pow(spectral_radius + 1.0, 2.0);
      gamma   = (3 - spectral_radius) / (2.0 * spectral_radius + 2.0);
      break;
    case GenAlphaType::GenAlpha:
      alpha_m = (2.0 * spectral_radius - 1.0) / (spectral_radius + 1.0);
      alpha_f = spectral_radius / (spectral_radius + 1.0);
      beta    = std::pow(1.0 - alpha_m + alpha_f, 2.0) / 4.0;
      gamma   = 0.5 - alpha_m + alpha_f;
      break;
    case GenAlphaType::HHTAlpha:
      alpha_m = 0.0;
      alpha_f = (1.0 - spectral_radius) / (spectral_radius + 1.0);
      beta    = std::pow(1.0 + alpha_f, 2.0) / 4.0;
      gamma   = 0.5 + alpha_f;
      break;
    case GenAlphaType::BossakAlpha:
      alpha_m = (spectral_radius - 1.0) / (spectral_radius + 1.0);
      alpha_f = 0.0;
      beta    = std::pow(1.0 - alpha_m, 2.0) / 4.0;
      gamma   = 0.5 - alpha_m;
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }
}

template<typename Number>
double
TimeIntGenAlphaBase<Number>::get_time_step_size() const
{
  return time_step;
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::set_current_time_step_size(double const & time_step_size)
{
  this->time_step = time_step_size;
}

template<typename Number>
double
TimeIntGenAlphaBase<Number>::get_scaling_factor_mass_from_acceleration() const
{
  return ((1.0 - alpha_m) / ((1.0 - alpha_f) * beta * std::pow(time_step, 2.0)));
}

template<typename Number>
double
TimeIntGenAlphaBase<Number>::get_scaling_factor_mass_from_velocity() const
{
  return (gamma / (beta * time_step));
}

template<typename Number>
double
TimeIntGenAlphaBase<Number>::get_mid_time() const
{
  return (this->time + (1.0 - alpha_f) * this->time_step);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::compute_const_vector_acceleration_remainder(
  VectorType &       const_vector,
  VectorType const & displacement_n,
  VectorType const & velocity_n,
  VectorType const & acceleration_n) const
{
  double const factor_dis = -get_scaling_factor_mass_from_acceleration();
  double const factor_vel = -(1.0 - alpha_m) / (beta * time_step);
  double const factor_acc = -(1.0 - alpha_m - 2.0 * beta) / (2.0 * beta);

  const_vector.equ(factor_dis, displacement_n);
  const_vector.add(factor_vel, velocity_n);
  const_vector.add(factor_acc, acceleration_n);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::compute_const_vector_velocity_remainder(
  VectorType &       const_vector,
  VectorType const & displacement_n,
  VectorType const & velocity_n,
  VectorType const & acceleration_n) const
{
  double const factor_dis = -get_scaling_factor_mass_from_velocity();
  double const factor_vel = alpha_f - (1.0 - alpha_f) * (gamma - beta) / beta;
  double const factor_acc = -(1.0 - alpha_f) * (gamma - 2.0 * beta) / (2.0 * beta) * time_step;

  const_vector.equ(factor_dis, displacement_n);
  const_vector.add(factor_vel, velocity_n);
  const_vector.add(factor_acc, acceleration_n);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::update_displacement(VectorType &       displacement_np,
                                                 VectorType const & displacement_n) const
{
  displacement_np.add(-alpha_f, displacement_n);
  displacement_np *= 1.0 / (1.0 - alpha_f);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::update_velocity(VectorType &       velocity_np,
                                             VectorType const & displacement_np,
                                             VectorType const & displacement_n,
                                             VectorType const & velocity_n,
                                             VectorType const & acceleration_n) const
{
  velocity_np.equ(gamma / (beta * time_step), displacement_np);
  velocity_np.add(-gamma / (beta * time_step), displacement_n);
  velocity_np.add(-(gamma - beta) / beta, velocity_n);
  velocity_np.add(-(gamma - 2.0 * beta) * time_step / (2.0 * beta), acceleration_n);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::update_acceleration(VectorType &       acceleration_np,
                                                 VectorType const & displacement_np,
                                                 VectorType const & displacement_n,
                                                 VectorType const & velocity_n,
                                                 VectorType const & acceleration_n) const
{
  acceleration_np.equ(1.0 / (beta * std::pow(time_step, 2.0)), displacement_np);
  acceleration_np.add(-1.0 / (beta * std::pow(time_step, 2.0)), displacement_n);
  acceleration_np.add(-1.0 / (beta * time_step), velocity_n);
  acceleration_np.add(-(1.0 - 2.0 * beta) / (2.0 * beta), acceleration_n);
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::do_timestep_pre_solve(bool const print_header)
{
  if(this->print_solver_info() and print_header)
    this->output_solver_info_header();
}

template<typename Number>
void
TimeIntGenAlphaBase<Number>::do_timestep_post_solve()
{
  prepare_vectors_for_next_timestep();

  // increment time
  this->time += time_step;
  ++this->time_step_number;

  // restart
  if(this->restart_data.write_restart == true)
  {
    this->write_restart();
  }

  if(this->print_solver_info())
  {
    this->output_remaining_time();
  }
}

template class TimeIntGenAlphaBase<float>;
template class TimeIntGenAlphaBase<double>;

} // namespace ExaDG
