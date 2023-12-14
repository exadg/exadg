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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_ENUM_TYPES_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_ENUM_TYPES_H_
#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
namespace Acoustics
{
enum class Formulation
{
  Undefined,
  /** Weak form of pressure gradient and velocity divergence terms. */
  Weak,
  /** Strong form of pressure gradient and velocity divergence terms. */
  Strong,
  /** Strong form of the pressure gradient term and weak form of the velocity divergence term.
   *  This way, the gradient is only used on scalar variables which reduces the number of sum
   *  factorization sweeps in the cell loop. Additionally, only the skew-symmetric formulation
   *  mathematically guarantees that energy will be non-increasing if the numerical quadrature
   *  is not exact (non-affine cells).
   */
  SkewSymmetric
};

enum class FluxFormulation
{
  Undefined,
  /** Standard local Lax-Friedrichs flux. */
  LaxFriedrichs,
  /** Similar to local Lax-Friedrichs flux, but adds dissipation only on the component normal
   *  component.
   */
  RoeType
};

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified,
  CFL
};

} // namespace Acoustics
} // namespace ExaDG

#endif /*EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_ENUM_TYPES_H_*/
