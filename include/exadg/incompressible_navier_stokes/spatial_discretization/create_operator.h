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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CREATE_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CREATE_OPERATOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>

namespace ExaDG
{
namespace IncNS
{
/**
 * Creates spatial discretization operator depending on type of solution strategy.
 */
template<int dim, typename Number>
std::shared_ptr<SpatialOperatorBase<dim, Number>>
create_operator(std::shared_ptr<Grid<dim, Number> const>        grid,
                unsigned int const                              degree_velocity,
                std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity,
                std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure,
                std::shared_ptr<FieldFunctions<dim>> const      field_functions,
                InputParameters const &                         parameters,
                std::string const &                             field,
                MPI_Comm const &                                mpi_comm)
{
  std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator;

  // initialize pde_operator
  if(parameters.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    pde_operator = std::make_shared<OperatorCoupled<dim, Number>>(grid,
                                                                  degree_velocity,
                                                                  boundary_descriptor_velocity,
                                                                  boundary_descriptor_pressure,
                                                                  field_functions,
                                                                  parameters,
                                                                  field,
                                                                  mpi_comm);
  }
  else if(parameters.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    pde_operator =
      std::make_shared<OperatorDualSplitting<dim, Number>>(grid,
                                                           degree_velocity,
                                                           boundary_descriptor_velocity,
                                                           boundary_descriptor_pressure,
                                                           field_functions,
                                                           parameters,
                                                           field,
                                                           mpi_comm);
  }
  else if(parameters.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    pde_operator =
      std::make_shared<OperatorPressureCorrection<dim, Number>>(grid,
                                                                degree_velocity,
                                                                boundary_descriptor_velocity,
                                                                boundary_descriptor_pressure,
                                                                field_functions,
                                                                parameters,
                                                                field,
                                                                mpi_comm);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return pde_operator;
}

} // namespace IncNS
} // namespace ExaDG



#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CREATE_OPERATOR_H_ */
