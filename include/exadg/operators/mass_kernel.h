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

#ifndef INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MassKernel
{
public:
  MassKernel()
  {
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(true, false, false);
    flags.cell_integrate = CellFlags(true, false, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_JxW_values;

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    get_volume_flux(double scaling_factor, T const & value) const
  {
    return scaling_factor * value;
  }
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_ */
