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

#ifndef INCLUDE_OPERATORS_MASS_OPERATOR_H_
#define INCLUDE_OPERATORS_MASS_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
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
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MassOperatorData<dim> const &             data);

  void
  set_scaling_factor(Number const & number);

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const;

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const final;

  MassKernel<dim, Number> kernel;

  mutable double scaling_factor;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_MASS_OPERATOR_H_ */
