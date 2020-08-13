/*
 * nonlinear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_

#include "elasticity_operator_base.h"

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

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
  initialize(MatrixFree<dim, Number> const &   matrix_free,
             AffineConstraints<double> const & constraint_matrix,
             OperatorData<dim> const &         data) override;

  /*
   * Evaluates the non-linear operator.
   */
  void
  evaluate_nonlinear(VectorType & dst, VectorType const & src) const;

  /*
   * Linearized operator:
   */
  void
  set_solution_linearization(VectorType const & vector) const;

  VectorType const &
  get_solution_linearization() const;

private:
  /*
   * Non-linear operator.
   */
  void
  reinit_cell_nonlinear(IntegratorCell & integrator, unsigned int const cell) const;

  void
  cell_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                    dst,
                      VectorType const &              src,
                      Range const &                   range) const;

  void
  face_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                    dst,
                      VectorType const &              src,
                      Range const &                   range) const;

  void
  boundary_face_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   range) const;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * d_h)_Omega + (Grad(v_h), P_h)_Omega
   *
   * with 1st Piola-Kirchhoff stress tensor P_h
   *
   *  P_h = F_h * S_h ,
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
  do_cell_integral_nonlinear(IntegratorCell & integrator) const;

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
                                  types::boundary_id const & boundary_id) const override;

  /*
   * Linearized operator.
   */
  void
  reinit_cell(unsigned int const cell) const override;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * delta d_h)_Omega + (Grad(v_h), delta P_h)_Omega
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
  do_cell_integral(IntegratorCell & integrator) const override;

  mutable std::shared_ptr<IntegratorCell> integrator_lin;
  mutable VectorType                      displacement_lin;
};

} // namespace Structure
} // namespace ExaDG

#endif
