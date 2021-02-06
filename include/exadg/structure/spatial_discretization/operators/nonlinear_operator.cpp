/*
 * nonlinear_operator.cpp
 *
 *  Created on: 18.03.2020
 *      Author: fehn
 */

#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::initialize(MatrixFree<dim, Number> const &   matrix_free,
                                           AffineConstraints<Number> const & constraint_matrix,
                                           OperatorData<dim> const &         data)
{
  Base::initialize(matrix_free, constraint_matrix, data);

  integrator_lin.reset(new IntegratorCell(*this->matrix_free));
  this->matrix_free->initialize_dof_vector(displacement_lin, data.dof_index);
  displacement_lin.update_ghost_values();
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_nonlinear(VectorType & dst, VectorType const & src) const
{
  this->matrix_free->loop(&This::cell_loop_nonlinear,
                          &This::face_loop_nonlinear,
                          &This::boundary_face_loop_nonlinear,
                          this,
                          dst,
                          src,
                          true);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_solution_linearization(VectorType const & vector) const
{
  displacement_lin = vector;
  displacement_lin.update_ghost_values();
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::VectorType const &
NonLinearOperator<dim, Number>::get_solution_linearization() const
{
  return displacement_lin;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_nonlinear(IntegratorCell &   integrator,
                                                      unsigned int const cell) const
{
  integrator.reinit(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                                                    VectorType &                    dst,
                                                    VectorType const &              src,
                                                    Range const &                   range) const
{
  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index,
                            this->operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator, cell);

    integrator.read_dof_values_plain(src);
    integrator.evaluate(this->operator_data.unsteady, true, false);

    do_cell_integral_nonlinear(integrator);

    integrator.integrate_scatter(this->operator_data.unsteady, true, dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::face_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                                                    VectorType &                    dst,
                                                    VectorType const &              src,
                                                    Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::boundary_face_loop_nonlinear(
  MatrixFree<dim, Number> const & matrix_free,
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
    // We write the integrator flags explicitly in this case since they
    // depend on the parameter pull_back_traction.
    if(this->operator_data.pull_back_traction)
    {
      this->integrator_m->read_dof_values_plain(src);
      this->integrator_m->evaluate(false, true);
    }

    do_boundary_integral_continuous(*this->integrator_m, matrix_free.get_boundary_id(face));

    this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral_nonlinear(IntegratorCell & integrator) const
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
    tensor const E = get_E<dim, Number>(F);

    // 2. Piola-Kirchhoff stresses
    tensor const S = material->evaluate_stress(E, integrator.get_current_cell_index(), q);

    // 1st Piola-Kirchhoff stresses P = F * S
    tensor const P = F * S;

    // Grad_v : P
    integrator.submit_gradient(P, q);

    if(this->operator_data.unsteady)
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &           integrator_m,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    auto traction = calculate_neumann_value<dim, Number>(
      q, integrator_m, boundary_type, boundary_id, this->operator_data.bc, this->time);

    if(this->operator_data.pull_back_traction)
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

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  integrator_lin->reinit(cell);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
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
    tensor const E_lin = get_E<dim, Number>(F_lin);

    // 2nd Piola-Kirchhoff stresses
    tensor const S_lin = material->evaluate_stress(E_lin, integrator.get_current_cell_index(), q);

    // directional derivative of 1st Piola-Kirchhoff stresses P

    // 1. elastic and initial displacement stiffness contributions
    tensor delta_P =
      F_lin *
      material->apply_C(transpose(F_lin) * Grad_delta, integrator.get_current_cell_index(), q);

    // 2. geometric (or initial stress) stiffness contribution
    delta_P += Grad_delta * S_lin;

    // Grad_v : delta_P
    integrator.submit_gradient(delta_P, q);

    if(this->operator_data.unsteady)
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
  }
}

template class NonLinearOperator<2, float>;
template class NonLinearOperator<2, double>;

template class NonLinearOperator<3, float>;
template class NonLinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
