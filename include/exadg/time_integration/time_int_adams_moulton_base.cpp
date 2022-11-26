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
#include <exadg/time_integration/time_int_adams_moulton_base.h>

namespace ExaDG
{
TimeIntAdamsMoultonBase::TimeIntAdamsMoultonBase(double const        start_time_,
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
    am(order_, start_with_low_order_),
    bdf(order_, start_with_low_order_)
{
}

void
TimeIntAdamsMoultonBase::update_time_integrator_constants()
{
  am.update(this->time_step_number, this->time_steps, this->adaptive_time_stepping);
  bdf.update(this->time_step_number, this->time_steps, this->adaptive_time_stepping);
}

} // namespace ExaDG
