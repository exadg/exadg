/*
 * mass_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
using namespace dealii;

template<int dim>
struct MassOperatorData : public OperatorBaseData
{
  MassOperatorData() : OperatorBaseData()
  {
  }
};

template<int dim, int n_components, typename Number>
class MassOperator : public OperatorBase<dim, Number, n_components>
{
public:
  typedef Number value_type;

  typedef OperatorBase<dim, Number, n_components> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;

  MassOperator();

  void
  initialize(MatrixFree<dim, Number> const &   matrix_free,
             AffineConstraints<Number> const & constraint_matrix,
             MassOperatorData<dim> const &     data);

  void
  set_scaling_factor(Number const & number);

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const;

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  MassKernel<dim, Number> kernel;

  mutable double scaling_factor;
};

} // namespace ExaDG

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
