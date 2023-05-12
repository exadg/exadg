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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_

#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

namespace ExaDG
{
namespace IncNS
{
/**
 * Creates time integrator depending on type of solution strategy.
 */
template<int dim, typename Number>
std::shared_ptr<TimeIntBDF<dim, Number>>
create_time_integrator(std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator,
                       std::shared_ptr<HelpersALE<Number> const>         helpers_ale,
                       Parameters const &                                parameters,
                       MPI_Comm const &                                  mpi_comm,
                       bool const                                        is_test,
                       std::shared_ptr<PostProcessorInterface<Number>>   postprocessor)
{
  std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator;

  if(parameters.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    std::shared_ptr<OperatorCoupled<dim, Number>> operator_coupled =
      std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(pde_operator);

    time_integrator = std::make_shared<IncNS::TimeIntBDFCoupled<dim, Number>>(
      operator_coupled, helpers_ale, parameters, mpi_comm, is_test, postprocessor);
  }
  else if(parameters.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    std::shared_ptr<OperatorDualSplitting<dim, Number>> operator_dual_splitting =
      std::dynamic_pointer_cast<OperatorDualSplitting<dim, Number>>(pde_operator);

    time_integrator = std::make_shared<IncNS::TimeIntBDFDualSplitting<dim, Number>>(
      operator_dual_splitting, helpers_ale, parameters, mpi_comm, is_test, postprocessor);
  }
  else if(parameters.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    std::shared_ptr<OperatorPressureCorrection<dim, Number>> operator_pressure_correction =
      std::dynamic_pointer_cast<OperatorPressureCorrection<dim, Number>>(pde_operator);

    time_integrator = std::make_shared<IncNS::TimeIntBDFPressureCorrection<dim, Number>>(
      operator_pressure_correction, helpers_ale, parameters, mpi_comm, is_test, postprocessor);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return time_integrator;
}

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_CREATE_TIME_INTEGRATOR_H_ */
