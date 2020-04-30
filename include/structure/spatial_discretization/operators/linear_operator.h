/*
 * linear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_

#include "elasticity_operator_base.h"

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
  do_cell_integral(IntegratorCell & integrator) const override
  {
    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // engineering strains (material tensor is symmetric)
      tensor const gradient = integrator.get_gradient(q);
      auto const   eps      = tensor_to_vector<dim, Number>(gradient);

      // Cauchy stresses
      material->reinit(eps);
      auto const   C     = material->get_dSdE();
      tensor const sigma = vector_to_tensor<dim, Number>(C * eps);

      // test with gradients
      integrator.submit_gradient(sigma, q);

      if(this->operator_data.unsteady)
        integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                  integrator.get_value(q),
                                q);
    }
  }

  /*
   * Computes Neumann BC integral
   *
   *  - (v_h, t)_{Gamma_N}
   */
  void
  do_boundary_integral_continuous(IntegratorFace &           integrator_m,
                                  types::boundary_id const & boundary_id) const override
  {
    BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      auto const neumann_value = calculate_neumann_value<dim, Number>(
        q, integrator_m, boundary_type, boundary_id, this->operator_data.bc, this->time);

      integrator_m.submit_value(-neumann_value, q);
    }
  }
};
} // namespace Structure

#endif
