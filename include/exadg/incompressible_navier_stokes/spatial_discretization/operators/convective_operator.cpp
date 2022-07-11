/*
 * convective_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_copy(VectorType const & src) const
{
  kernel->set_velocity_copy(src);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_ptr(VectorType const & src) const
{
  kernel->set_velocity_ptr(src);
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel->get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel)
{
  operator_data = data;

  kernel = convective_kernel;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator(VectorType &       dst,
                                                             VectorType const & src,
                                                             Number const       time) const
{
  this->time = time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          true /*zero_dst_vector = true*/,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator_add(VectorType &       dst,
                                                                 VectorType const & src,
                                                                 Number const       time) const
{
  this->time = time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          false /*zero_dst_vector = false*/,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function rhs() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function rhs_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  IntegratorCell integrator(matrix_free,
                            operator_data.dof_index,
                            operator_data.quad_index_nonlinear);
  IntegratorCell integrator_grid_velocity(matrix_free,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator.gather_evaluate(src, this->integrator_flags.cell_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(cell);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_cell_integral_nonlinear_operator(integrator, integrator_grid_velocity);

    integrator.integrate_scatter(this->integrator_flags.cell_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::face_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  IntegratorFace integrator_m(matrix_free,
                              true,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);
  IntegratorFace integrator_p(matrix_free,
                              false,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);

  IntegratorFace integrator_grid_velocity(matrix_free,
                                          true,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

    integrator_p.gather_evaluate(src, this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(face);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_face_integral_nonlinear_operator(integrator_m, integrator_p, integrator_grid_velocity);

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);

    integrator_p.integrate_scatter(this->integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::boundary_face_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  IntegratorFace integrator_m(matrix_free,
                              true,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);

  IntegratorFace integrator_grid_velocity(matrix_free,
                                          true,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(face);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_boundary_integral_nonlinear_operator(integrator_m,
                                            integrator_grid_velocity,
                                            matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_nonlinear_operator(
  IntegratorCell & integrator,
  IntegratorCell & integrator_u_grid) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u = integrator.get_value(q);

    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F = outer_product(u, u);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: (u * grad) u = grad(u) * u
      tensor gradient_u = integrator.get_gradient(q);
      if(operator_data.kernel_data.ale == true)
        u -= integrator_u_grid.get_value(q);

      vector F = gradient_u * u;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral_nonlinear_operator(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p,
  IntegratorFace & integrator_grid_velocity) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector u_grid;
    if(operator_data.kernel_data.ale == true)
      u_grid = integrator_grid_velocity.get_value(q);

    std::tuple<vector, vector> flux =
      kernel->calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m, u_grid);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_nonlinear_operator(
  IntegratorFace &                   integrator,
  IntegratorFace &                   integrator_grid_velocity,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = calculate_exterior_value_nonlinear(u_m,
                                                    q,
                                                    integrator,
                                                    boundary_type,
                                                    operator_data.kernel_data.type_dirichlet_bc,
                                                    boundary_id,
                                                    operator_data.bc,
                                                    this->time);

    vector normal_m = integrator.get_normal_vector(q);

    vector u_grid;
    if(operator_data.kernel_data.ale == true)
      u_grid = integrator_grid_velocity.get_value(q);

    vector flux =
      kernel->calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, u_grid, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel->reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel->reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based(
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel->reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector delta_u = integrator.get_value(q);

    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      tensor flux = kernel->get_volume_flux_linearized_divergence_formulation(delta_u, q);

      integrator.submit_gradient(flux, q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      tensor grad_delta_u = integrator.get_gradient(q);

      vector flux =
        kernel->get_volume_flux_linearized_convective_formulation(delta_u, grad_delta_u, q);

      integrator.submit_value(flux, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux = kernel->calculate_flux_linearized_interior_and_neighbor(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(std::get<0>(flux) /* flux_m */, q);
    integrator_p.submit_value(std::get<1>(flux) /* flux_p */, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    // TODO
    // Accessing exterior data is currently not available in deal.II/matrixfree.
    // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
    // are not calculated exactly.
    vector u_p = u_m;

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m; // set exterior value to zero
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_p = -integrator_p.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_p, u_m, delta_u_p, delta_u_m, normal_p, q);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  // make sure that this function is only accessed for OperatorType::homogeneous
  AssertThrow(
    operator_type == OperatorType::homogeneous,
    dealii::ExcMessage(
      "For the linearized convective operator, only OperatorType::homogeneous makes sense."));

  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = calculate_exterior_value_nonlinear(u_m,
                                                    q,
                                                    integrator,
                                                    boundary_type,
                                                    operator_data.kernel_data.type_dirichlet_bc,
                                                    boundary_id,
                                                    operator_data.bc,
                                                    this->time);

    vector delta_u_m = integrator.get_value(q);
    vector delta_u_p =
      kernel->calculate_exterior_value_linearized(delta_u_m, q, integrator, boundary_type);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel->calculate_flux_linearized_boundary(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, boundary_type, q);

    integrator.submit_value(flux, q);
  }
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
