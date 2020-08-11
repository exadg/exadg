/*
 * mass_matrix_operator.h
 *
 *  Created on: Jun 27, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include "../matrix_free/integrators.h"

#include "integrator_flags.h"
#include "mapping_flags.h"

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


#endif /* INCLUDE_OPERATORS_MASS_MATRIX_OPERATOR_H_ */
