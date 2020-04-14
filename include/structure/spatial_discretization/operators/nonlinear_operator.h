/*
 * nonlinear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_

#include "../../material/material_handler.h"

#include "continuum_mechanics_util.h"
#include "operator_data.h"
#include "tensor_util.h"

namespace Structure
{
template<int dim, typename Number>
class NonLinearOperator : public OperatorBase<dim, Number, OperatorData<dim>, dim>
{
public:
  typedef NonLinearOperator<dim, Number>                    This;
  typedef OperatorBase<dim, Number, OperatorData<dim>, dim> Base;
  typedef CellIntegrator<dim, dim, Number>                  IntegratorCell;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>      Range;

  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         OperatorData<dim> const &         operator_data)
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    integrator_lin.reset(new IntegratorCell(*this->matrix_free));
    this->matrix_free->initialize_dof_vector(displacement_lin, operator_data.dof_index);
    displacement_lin.update_ghost_values();

    this->integrator_flags = this->get_integrator_flags();

    material_handler.initialize(this->get_data().material_descriptor);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    // evaluation of Neumann BCs
    flags.face_evaluate  = FaceFlags(false, false);
    flags.face_integrate = FaceFlags(true, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_gradients | update_JxW_values;

    flags.boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  /*
   * Evaluates the non-linear operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src) const
  {
    this->matrix_free->cell_loop(&This::cell_loop_nonlinear, this, dst, src, true);
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
  cell_loop_nonlinear(MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                    dst,
                      VectorType const &              src,
                      Range const &                   range) const
  {
    IntegratorCell integrator(matrix_free, this->data.dof_index, this->data.quad_index);

    for(auto cell = range.first; cell < range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values_plain(src);
      integrator.evaluate(false, true, false);

      material_handler.reinit(*this->matrix_free, cell);

      do_cell_integral_nonlinear(integrator);

      integrator.integrate_scatter(false, true, dst);
    }
  }

  void
  do_cell_integral_nonlinear(IntegratorCell & integrator) const
  {
    std::shared_ptr<Material<dim, Number>> material = material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // kinematics
      tensor const Grad_d = integrator.get_gradient(q);
      tensor const F      = get_F<dim, Number>(Grad_d); // material deformation gradient

      // strains and stresses
      auto const E = get_E<dim, Number>(F); // Green-Lagrange strains
      material->reinit(E);
      auto const S = material->get_S(); // 2. Piola-Kirchhoff stresses

      // 1st Piola-Kirchhoff stresses P = F * S
      tensor P = F * vector_to_tensor<dim, Number>(S);

      // TODO updated Lagrangian formulation
      //      if(this->data.updated_formulation)
      //        P = vector_to_tensor<dim, Number>(get_sigma<dim, Number>(S, F));

      // Grad_v : P
      integrator.submit_gradient(P, q);
    }
  }

  /*
   * Linearized operator.
   */
  void
  reinit_cell(unsigned int const cell) const
  {
    Base::reinit_cell(cell);

    integrator_lin->reinit(cell);

    material_handler.reinit(*this->matrix_free, cell);
  }

  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    integrator_lin->read_dof_values_plain(displacement_lin);
    integrator_lin->evaluate(false, true);

    std::shared_ptr<Material<dim, Number>> material = material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // kinematics
      tensor const Grad_delta = integrator.get_gradient(q);

      tensor const F_lin = get_F<dim, Number>(integrator_lin->get_gradient(q));

      // strains and stresses
      auto const E_lin = get_E<dim, Number>(F_lin);
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

      // TODO: updated Lagrangian formulation
      //      if(this->data.updated_formulation)
      //        delta_P += Grad_du * vector_to_tensor<dim, Number>(get_sigma<dim, Number>(S_lin,
      //        F_lin));

      // test with gradients
      integrator.submit_gradient(delta_P, q);
    }
  }

  mutable MaterialHandler<dim, Number> material_handler;

  mutable std::shared_ptr<IntegratorCell> integrator_lin;
  mutable VectorType                      displacement_lin;
};

} // namespace Structure

#endif
