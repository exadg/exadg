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

#include <exadg/time_integration/time_int_bdf_base.h>

namespace ExaDG
{
TimeIntBDFBase::TimeIntBDFBase(double const        start_time_,
                               double const        end_time_,
                               unsigned int const  max_number_of_time_steps_,
                               unsigned const      order_,
                               bool const          start_with_low_order_,
                               bool const          adaptive_time_stepping_,
                               RestartData const & restart_data_,
                               MPI_Comm const &    mpi_comm_,
                               bool const          is_test_)
  : TimeIntMultistepBase(start_time_,
                         end_time_,
                         max_number_of_time_steps_,
                         order_,
                         start_with_low_order_,
                         adaptive_time_stepping_,
                         restart_data_,
                         mpi_comm_,
                         is_test_),
    bdf(order_, start_with_low_order_),
    extra(order_, start_with_low_order_)
{
}



double
TimeIntBDFBase::get_scaling_factor_time_derivative_term() const
{
  return bdf.get_gamma0() / time_steps[0];
}

void
TimeIntBDFBase::update_time_integrator_constants()
{
  bdf.update(time_step_number, adaptive_time_stepping, time_steps);
  extra.update(time_step_number, adaptive_time_stepping, time_steps);

  // use this function to check the correctness of the time integrator constants
  //  std::cout << std::endl << "Time step " << time_step_number << std::endl << std::endl;
  //  std::cout << "Coefficients BDF time integration scheme:" << std::endl;
  //  bdf.print();
  //  std::cout << "Coefficients extrapolation scheme:" << std::endl;
  //  extra.print();
}
} // namespace ExaDG
