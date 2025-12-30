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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_MASS_KERNEL_H_
#define EXADG_OPERATORS_MASS_KERNEL_H_

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
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

    flags.cell_evaluate  = dealii::EvaluationFlags::values;
    flags.cell_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values;

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

  /*
   * Variable coefficients not managed by this class.
   */
  void
  set_variable_coefficients_ptr(
    VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients_in)
  {
    variable_coefficients = variable_coefficients_in;
  }

  VariableCoefficients<dealii::VectorizedArray<Number>> const *
  get_variable_coefficients_ptr() const
  {
    return variable_coefficients;
  }

private:
  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_MASS_KERNEL_H_ */
