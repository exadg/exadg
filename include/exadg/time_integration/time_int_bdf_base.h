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

#ifndef EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_
#define EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_

// ExaDG
#include <exadg/time_integration/bdf_constants.h>
#include <exadg/time_integration/extrapolation_constants.h>
#include <exadg/time_integration/time_int_multistep_base.h>

namespace ExaDG
{
class TimeIntBDFBase : public TimeIntMultistepBase
{
public:
  TimeIntBDFBase(double const        start_time_,
                 double const        end_time_,
                 unsigned int const  max_number_of_time_steps_,
                 unsigned const      order_,
                 bool const          start_with_low_order_,
                 bool const          adaptive_time_stepping_,
                 RestartData const & restart_data_,
                 MPI_Comm const &    mpi_comm_,
                 bool const          is_test_);

protected:
  double
  get_scaling_factor_time_derivative_term() const;

  void
  update_time_integrator_constants() override;

  /*
   * Time integration constants. The extrapolation scheme is not necessarily used for a BDF time
   * integration scheme with fully implicit time stepping, implying a violation of the Liskov
   * substitution principle (OO software design principle). However, it does not appear to be
   * reasonable to complicate the inheritance due to this fact.
   */
  BDFTimeIntegratorConstants bdf;
  ExtrapolationConstants     extra;
};
} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_ */
