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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_

#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>

namespace ExaDG
{
namespace ConvDiff
{
/**
 * Creates time integrator depending on type of time integration strategy.
 */
template<int dim, typename Number>
std::shared_ptr<TimeIntBase>
create_time_integrator(std::shared_ptr<Operator<dim, Number>>          pde_operator,
                       Parameters const &                              parameters,
                       unsigned int const                              refine_steps_time,
                       MPI_Comm const &                                mpi_comm,
                       bool const                                      is_test,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor)
{
  std::shared_ptr<TimeIntBase> time_integrator;

  if(parameters.temporal_discretization == TemporalDiscretization::ExplRK)
  {
    time_integrator = std::make_shared<TimeIntExplRK<Number>>(
      pde_operator, parameters, refine_steps_time, mpi_comm, is_test, postprocessor);
  }
  else if(parameters.temporal_discretization == TemporalDiscretization::BDF)
  {
    time_integrator = std::make_shared<TimeIntBDF<dim, Number>>(
      pde_operator, parameters, refine_steps_time, mpi_comm, is_test, postprocessor);
  }
  else
  {
    AssertThrow(parameters.temporal_discretization == TemporalDiscretization::ExplRK ||
                  parameters.temporal_discretization == TemporalDiscretization::BDF,
                ExcMessage("Specified time integration scheme is not implemented!"));
  }

  return time_integrator;
}

} // namespace ConvDiff
} // namespace ExaDG



#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_ */
