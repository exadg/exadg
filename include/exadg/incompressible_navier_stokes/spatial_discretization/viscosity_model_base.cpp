/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_base.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
ViscosityModelBase<dim, Number>::ViscosityModelBase()
  : dealii::Subscriptor(), dof_index_velocity(0), matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
ViscosityModelBase<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
  unsigned int const                                     dof_index_velocity_in)
{
  matrix_free    = &matrix_free_in;
  viscous_kernel = viscous_kernel_in;

  dof_index_velocity = dof_index_velocity_in;
}

template class ViscosityModelBase<2, float>;
template class ViscosityModelBase<2, double>;
template class ViscosityModelBase<3, float>;
template class ViscosityModelBase<3, double>;

} // namespace IncNS
} // namespace ExaDG
