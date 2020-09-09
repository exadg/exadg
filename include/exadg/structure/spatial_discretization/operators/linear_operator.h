/*
 * linear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_

#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
class LinearOperator : public ElasticityOperatorBase<dim, Number>
{
private:
  typedef ElasticityOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * d_h)_Omega + (grad(v_h), sigma_h)_Omega
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
   * Computes Neumann BC integral
   *
   *  - (v_h, t)_{Gamma_N}
   */
  void
  do_boundary_integral_continuous(IntegratorFace &           integrator_m,
                                  types::boundary_id const & boundary_id) const override;
};

} // namespace Structure
} // namespace ExaDG

#endif
