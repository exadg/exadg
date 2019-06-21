/*
 * mass_matrix_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/operator_base.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
template<int dim, typename Number>
class MassMatrixKernel
{
public:
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  MassMatrixKernel() : scaling_factor(1.0)
  {
  }

  void
  reinit(double const & factor) const
  {
    set_scaling_factor(factor);
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

  Number
  get_scaling_factor() const
  {
    return scaling_factor;
  }

  void
  set_scaling_factor(Number const & number) const
  {
    scaling_factor = number;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(vector const & value) const
  {
    return scaling_factor * value;
  }

private:
  mutable Number scaling_factor;
};

} // namespace Operators

struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData, dim>
{
public:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::Range          Range;
  typedef typename Base::IntegratorCell IntegratorCell;

  void
  set_scaling_factor(Number const & number) const;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         MassMatrixOperatorData const &    operator_data) const;

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    kernel.set_scaling_factor(factor);

    this->apply(dst, src);

    kernel.set_scaling_factor(1.0);
  }

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    kernel.set_scaling_factor(factor);

    this->apply_add(dst, src);

    kernel.set_scaling_factor(1.0);
  }

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  Operators::MassMatrixKernel<dim, Number> kernel;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
