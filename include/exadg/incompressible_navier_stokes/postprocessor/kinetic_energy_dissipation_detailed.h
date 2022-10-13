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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/postprocessor/kinetic_energy_calculation.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
class KineticEnergyCalculatorDetailed : public KineticEnergyCalculator<dim, Number>
{
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef KineticEnergyCalculator<dim, Number> Base;

  typedef SpatialOperatorBase<dim, Number> NavierStokesOperator;

public:
  KineticEnergyCalculatorDetailed(MPI_Comm const & comm);

  void
  setup(NavierStokesOperator const &            navier_stokes_operator_in,
        dealii::MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const                      dof_index_in,
        unsigned int const                      quad_index_in,
        KineticEnergyData const &               kinetic_energy_data_in);

  void
  evaluate(VectorType const & velocity, double const time, bool const unsteady);

private:
  void
  calculate_detailed(VectorType const & velocity, double const time);

  dealii::SmartPointer<NavierStokesOperator const> navier_stokes_operator;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_ \
        */
