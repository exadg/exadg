/*
 * nonlinear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_

#include "elasticity_operator_base.h"

namespace Structure
{
template<int dim, typename Number>
class NonLinearOperator : public ElasticityOperatorBase<dim, Number>
{
private:
  typedef ElasticityOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef NonLinearOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

public:
  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         OperatorData<dim> const &         operator_data) override
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    integrator_lin.reset(new IntegratorCell(*this->matrix_free));
    this->matrix_free->initialize_dof_vector(displacement_lin, operator_data.dof_index);
    displacement_lin.update_ghost_values();
  }

  /*
   * Evaluates the non-linear operator.
   */
  void
  evaluate_nonlinear(VectorType & dst, VectorType const & src) const
  {
    this->matrix_free->loop(&This::cell_loop_nonlinear,
                            &This::face_loop_empty,
                            &This::boundary_face_loop_nonlinear,
                            this,
                            dst,
                            src,
                            true);
  }

  /*
   * linearized operator
   */
  void
  set_solution_linearization(VectorType const & vector) const
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();
  }

private:
  /*
   * Non-linear operator.
   */
  void
  reinit_cell_nonlinear(IntegratorCell & integrator, unsigned int const cell) const
  {
    integrator.reinit(cell);

    this->material_handler.reinit(*this->matrix_free, cell);
  }

  void
  cell_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                    dst,
                      VectorType const &              src,
                      Range const &                   range) const
  {
    IntegratorCell integrator(matrix_free, this->data.dof_index, this->data.quad_index);

    for(auto cell = range.first; cell < range.second; ++cell)
    {
      reinit_cell_nonlinear(integrator, cell);

      integrator.read_dof_values_plain(src);
      integrator.evaluate(false, true, false);

      do_cell_integral_nonlinear(integrator);

      integrator.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const
  {
    (void)matrix_free;
    (void)dst;
    (void)src;
    (void)range;
  }

  void
  boundary_face_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   range) const
  {
    (void)src;

    // apply Neumann BCs
    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(face);

      // In case of a pull-back of the traction vector, we need to evaluate
      // the displacement gradient to obtain the surface area ratio da/dA.
      // We explicitly write the integrator flags in this case since they
      // depend on the parameter pull_back_traction.
      if(this->data.pull_back_traction)
        this->integrator_m->gather_evaluate(src, false, true);

      do_boundary_integral_continuous(*this->integrator_m, matrix_free.get_boundary_id(face));

      this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                            this->integrator_flags.face_integrate.gradient,
                                            dst);
    }
  }

  /*
   * Calculates the integral
   *
   *  (Grad(v_h), P_h)_Omega
   *
   * with 1st Piola-Kirchhoff stress tensor P_h
   *
   *  P_h = F * S_h ,
   *
   * 2nd Piola-Kirchhoff stress tensor S_h
   *
   *  S_h = function(E_h) ,
   *
   * Green-Lagrange strain tensor E_h
   *
   *  E_h = 1/2 (F_h^T * F_h - 1) ,
   *
   * material deformation gradient F_h
   *
   *  F_h = 1 + Grad(d_h) ,
   *
   * where
   *
   *  d_h denotes the displacement vector.
   */
  void
  do_cell_integral_nonlinear(IntegratorCell & integrator) const
  {
    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material displacement gradient
      tensor const Grad_d = integrator.get_gradient(q);

      // material deformation gradient
      tensor const F = get_F<dim, Number>(Grad_d);

      // Green-Lagrange strains
      auto const E = get_E<dim, Number>(F);

      // 2. Piola-Kirchhoff stresses
      material->reinit(E);
      auto const S = material->get_S();

      // 1st Piola-Kirchhoff stresses P = F * S
      tensor P = F * vector_to_tensor<dim, Number>(S);

      // Grad_v : P
      integrator.submit_gradient(P, q);
    }
  }

  /*
   * Computes Neumann BC integral
   *
   *  - (v_h, t_0)_{Gamma_N}
   *
   * with traction
   *
   *  t_0 = da/dA t .
   *
   * If the traction is specified as force per surface area of the underformed
   * body, the specified traction t is interpreted as t_0 = t, and no pull-back
   * is necessary.
   */
  void
  do_boundary_integral_continuous(IntegratorFace &           integrator_m,
                                  types::boundary_id const & boundary_id) const
  {
    BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      auto traction = calculate_neumann_value<dim, Number>(
        q, integrator_m, boundary_type, boundary_id, this->data.bc, this->time);

      if(this->data.pull_back_traction)
      {
        tensor F = get_F<dim, Number>(integrator_m.get_gradient(q));
        vector N = integrator_m.get_normal_vector(q);
        // da/dA * n = det F F^{-T} * N := n_star
        // -> da/dA = n_star.norm()
        vector n_star = determinant(F) * transpose(invert(F)) * N;
        // t_0 = da/dA * t
        traction *= n_star.norm();
      }

      integrator_m.submit_value(-traction, q);
    }
  }

  /*
   * Linearized operator.
   */
  void
  reinit_cell(unsigned int const cell) const override
  {
    Base::reinit_cell(cell);

    integrator_lin->reinit(cell);
  }

  /*
   * Calculates the integral
   *
   *  (Grad(v_h), delta P_h)_Omega
   *
   * with the directional derivative of the 1st Piola-Kirchhoff stress tensor P_h
   *
   *  delta P_h = d(P)/d(d)|_{d_lin} * delta d_h ,
   *
   * with the point of linearization
   *
   *  d_lin ,
   *
   * and displacement increment
   *
   *  delta d_h .
   *
   * Computing the linearization yields
   *
   *  delta P_h = + Grad(delta_d) * S(d_lin)
   *              + F(d_lin) * (C_lin : (F^T(d_lin) * Grad(delta d))) .
   *
   *  Note that a dependency of the Neumann BC on the displacements d through
   *  the area ratio da/dA = function(d) is neglected in the linearization.
   */
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    integrator_lin->read_dof_values_plain(displacement_lin);
    integrator_lin->evaluate(false, true);

    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // kinematics
      tensor const Grad_delta = integrator.get_gradient(q);

      tensor const F_lin = get_F<dim, Number>(integrator_lin->get_gradient(q));

      // Green-Lagrange strains
      auto const E_lin = get_E<dim, Number>(F_lin);

      // 2nd Piola-Kirchhoff stresses and tangential stiffness matrix
      material->reinit(E_lin);
      auto const S_lin = material->get_S();
      auto const C_lin = material->get_dSdE();

      // directional derivative of 1st Piola-Kirchhoff stresses P
      tensor delta_P;

      // 1. elastic and initial displacement stiffness contributions
      delta_P = F_lin * vector_to_tensor<dim, Number>(
                          C_lin * tensor_to_vector<dim, Number>(transpose(F_lin) * Grad_delta));

      // 2. geometric (or initial stress) stiffness contribution
      delta_P += Grad_delta * vector_to_tensor<dim, Number>(S_lin);

      // Grad_v : delta_P
      integrator.submit_gradient(delta_P, q);
    }
  }

  mutable std::shared_ptr<IntegratorCell> integrator_lin;
  mutable VectorType                      displacement_lin;
};

} // namespace Structure

#endif
