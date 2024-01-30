/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2024 by the ExaDG authors
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

#ifndef EXADG_AERO_ACOUSTIC_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define EXADG_AERO_ACOUSTIC_USER_INTERFACE_FIELD_FUNCTIONS_H_

#include <exadg/utilities/optional_function.h>

namespace ExaDG
{
namespace AeroAcoustic
{
template<int dim>
struct FieldFunctions
{
  /*
   * The function source_term_blend_in is mainly used to blend in the source
   * term over time. In some rare cases it might also be beneficial to fade
   * out the source term in space: If large source term contributions are
   * placed at the boundary of the CFD domain, there can be problems when
   * interpolating to the non-matching acoustic domain (that is mostly
   * much larger than the CFD domain).
   *
   * If the source term is only used to blend in the source term over time
   * it has to be evaluated only once every time step. Therefore, we are using a
   * SpatialAwareFunction, which provides information if the function varies in
   * space for a given time.
   */
  std::shared_ptr<Utilities::SpatialAwareFunction<dim>> source_term_blend_in;
};

} // namespace AeroAcoustic
} // namespace ExaDG

#endif /* EXADG_AERO_ACOUSTIC_USER_INTERFACE_FIELD_FUNCTIONS_H_ */
