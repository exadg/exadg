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

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_

#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class LinearOperator : public ElasticityOperatorBase<dim, Number>
{
private:
  typedef ElasticityOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * density * d_h)_Omega + (grad(v_h), sigma_h)_Omega
   *
   * with
   *
   *  sigma_h = C : eps_h, eps_h = grad(d_h)
   *
   * where
   *
   *  d_h denotes the displacement vector.
   */
  void
  do_cell_integral(IntegratorCell & integrator) const override;

  /*
   * Computes boundary integrals
   *
   * inhomogeneous operator:
   *  - (v_h, t)_{Gamma_N} - (v_h, - p * N)_{Gamma_R}
   *
   * homogeneous operator:
   *  + (v_h, k * d_h + c * factor_velocity * d_h)_{Gamma_R}
   */
  void
  do_boundary_integral_continuous(IntegratorFace &                   integrator_m,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const override;
};

} // namespace Structure
} // namespace ExaDG

#endif
