/*
 * mass_matrix_operator.h
 *
 *  Created on: Jun 27, 2019
 *      Author: fehn
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
class MassMatrixKernel
{
public:
  MassMatrixKernel()
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
