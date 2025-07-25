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

#ifndef EXADG_OPERATORS_MASS_OPERATOR_H_
#define EXADG_OPERATORS_MASS_OPERATOR_H_

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
template<int dim, typename Number>
struct MassOperatorData : public OperatorBaseData
{
  MassOperatorData()
    : OperatorBaseData(),
      coefficient_is_variable(false),
      variable_coefficients(nullptr),
      consider_inverse_coefficient(false)
  {
  }

  // variable coefficients
  bool                                                          coefficient_is_variable;
  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;
  // use the inverse of the coefficients stored in `variable_coefficients`
  bool consider_inverse_coefficient;
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
             MassOperatorData<dim, Number> const &     data);

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

  // Variable coefficients not managed by this class.
  MassOperatorData<dim, Number> operator_data;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_MASS_OPERATOR_H_ */
