/*
 * convective_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "convective_operator.h"

namespace IncNS
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_solution_linearization(VectorType const & src) const
{
  kernel.set_velocity(src);
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_solution_linearization() const
{
  return kernel.get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ConvectiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  kernel.reinit(matrix_free,
                operator_data.kernel_data,
                operator_data.dof_index,
                operator_data.quad_index);

  this->integrator_flags = kernel.get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator(VectorType &       dst,
                                                             VectorType const & src,
                                                             Number const evaluation_time) const
{
  this->eval_time = evaluation_time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          true /*zero_dst_vector = true*/,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator_add(VectorType &       dst,
                                                                 VectorType const & src,
                                                                 Number const evaluation_time) const
{
  this->eval_time = evaluation_time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          false /*zero_dst_vector = false*/,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_linear_transport(
  VectorType &       dst,
  VectorType const & src,
  Number const       evaluation_time,
  VectorType const & velocity_transport) const
{
  set_solution_linearization(velocity_transport);

  this->eval_time = evaluation_time;

  this->matrix_free->loop(&This::cell_loop_linear_transport,
                          &This::face_loop_linear_transport,
                          &This::boundary_face_loop_linear_transport,
                          this,
                          dst,
                          src,
                          true /*zero_dst_vector = true*/,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              ExcMessage("The function rhs() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;

  AssertThrow(
    false, ExcMessage("The function rhs_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(
    false, ExcMessage("The function evaluate() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              ExcMessage(
                "The function evaluate_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_nonlinear_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   cell_range) const
{
  (void)matrix_free;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    this->integrator->reinit(cell);

    this->integrator->gather_evaluate(src,
                                      this->integrator_flags.cell_evaluate.value,
                                      this->integrator_flags.cell_evaluate.gradient,
                                      this->integrator_flags.cell_evaluate.hessian);

    do_cell_integral_nonlinear_operator(*this->integrator);

    this->integrator->integrate_scatter(this->integrator_flags.cell_integrate.value,
                                        this->integrator_flags.cell_integrate.gradient,
                                        dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::face_loop_nonlinear_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  (void)matrix_free;

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    this->integrator_m->reinit(face);
    this->integrator_p->reinit(face);

    this->integrator_m->gather_evaluate(src,
                                        this->integrator_flags.face_evaluate.value,
                                        this->integrator_flags.face_evaluate.gradient);

    this->integrator_p->gather_evaluate(src,
                                        this->integrator_flags.face_evaluate.value,
                                        this->integrator_flags.face_evaluate.gradient);

    do_face_integral_nonlinear_operator(*this->integrator_m, *this->integrator_p);

    this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);

    this->integrator_p->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::boundary_face_loop_nonlinear_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    this->integrator_m->reinit(face);

    this->integrator_m->gather_evaluate(src,
                                        this->integrator_flags.face_evaluate.value,
                                        this->integrator_flags.face_evaluate.gradient);

    do_boundary_integral_nonlinear_operator(*this->integrator_m, matrix_free.get_boundary_id(face));

    this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_nonlinear_operator(
  IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u = integrator.get_value(q);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F = outer_product(u, u);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: (u * grad) u = grad(u) * u
      tensor gradient_u = integrator.get_gradient(q);
      vector F          = gradient_u * u;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F          = outer_product(u, u);
      scalar divergence = integrator.get_divergence(q);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
      integrator.submit_value(-0.5 * divergence * u, q);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral_nonlinear_operator(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux =
      kernel.calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_nonlinear_operator(
  IntegratorFace &           integrator,
  types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = kernel.calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel.calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_linear_transport(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   cell_range) const
{
  (void)matrix_free;

  IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    this->integrator->reinit(cell);

    this->integrator->gather_evaluate(src,
                                      flags.cell_evaluate.value,
                                      flags.cell_evaluate.gradient,
                                      flags.cell_evaluate.hessian);

    kernel.reinit_cell_linear_transport(cell);

    do_cell_integral_linear_transport(*this->integrator);

    this->integrator->integrate_scatter(flags.cell_integrate.value,
                                        flags.cell_integrate.gradient,
                                        dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::face_loop_linear_transport(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  (void)matrix_free;

  IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    this->integrator_m->reinit(face);
    this->integrator_p->reinit(face);

    this->integrator_m->gather_evaluate(src,
                                        flags.face_evaluate.value,
                                        flags.face_evaluate.gradient);

    this->integrator_p->gather_evaluate(src,
                                        flags.face_evaluate.value,
                                        flags.face_evaluate.gradient);

    kernel.reinit_face_linear_transport(face);

    do_face_integral_linear_transport(*this->integrator_m, *this->integrator_p);

    this->integrator_m->integrate_scatter(flags.face_integrate.value,
                                          flags.face_integrate.gradient,
                                          dst);

    this->integrator_p->integrate_scatter(flags.face_integrate.value,
                                          flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::boundary_face_loop_linear_transport(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  IntegratorFlags flags = kernel.get_integrator_flags_linear_transport();

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    this->integrator_m->reinit(face);
    this->integrator_m->gather_evaluate(src,
                                        flags.face_evaluate.value,
                                        flags.face_evaluate.gradient);

    kernel.reinit_boundary_face_linear_transport(face);

    do_boundary_integral_linear_transport(*this->integrator_m, matrix_free.get_boundary_id(face));

    this->integrator_m->integrate_scatter(flags.face_integrate.value,
                                          flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_linear_transport(
  IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector w = kernel.get_velocity_cell(q);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F = uw
      vector u = integrator.get_value(q);
      tensor F = outer_product(u, w);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: grad(u) * w
      tensor grad_u = integrator.get_gradient(q);
      vector F      = grad_u * w;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral_linear_transport(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = integrator_m.get_value(q);
    vector u_p = integrator_p.get_value(q);

    vector w_m = kernel.get_velocity_m(q);
    vector w_p = kernel.get_velocity_p(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux =
      kernel.calculate_flux_linear_transport_interior_and_neighbor(u_m, u_p, w_m, w_p, normal_m);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_linear_transport(
  IntegratorFace &           integrator,
  types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = kernel.calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

    // concerning the transport velocity w, use the same value for interior and
    // exterior states, i.e., do not prescribe boundary conditions
    vector w_m = kernel.get_velocity_m(q);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux =
      kernel.calculate_flux_linear_transport_boundary(u_m, u_p, w_m, w_m, normal_m, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  kernel.reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel.reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel.reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel.reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector delta_u = integrator.get_value(q);
    vector u       = kernel.get_velocity_cell(q);

    if(this->operator_data.kernel_data.formulation ==
       FormulationConvectiveTerm::DivergenceFormulation)
    {
      tensor flux = kernel.get_volume_flux_linearized_divergence_formulation(u, delta_u);

      integrator.submit_gradient(flux, q);
    }
    else if(this->operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      tensor grad_u       = kernel.get_velocity_gradient_cell(q);
      tensor grad_delta_u = integrator.get_gradient(q);

      vector flux =
        kernel.get_volume_flux_linearized_convective_formulation(u, delta_u, grad_u, grad_delta_u);

      integrator.submit_value(flux, q);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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
    vector u_m = kernel.get_velocity_m(q);
    vector u_p = kernel.get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux = kernel.calculate_flux_linearized_interior_and_neighbor(
      u_m, u_p, delta_u_m, delta_u_p, normal_m);

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
    vector u_m = kernel.get_velocity_m(q);
    vector u_p = kernel.get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel.calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

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
    vector u_m = kernel.get_velocity_m(q);
    // TODO
    // Accessing exterior data is currently not available in deal.II/matrixfree.
    // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
    // are not calculated exactly.
    vector u_p = u_m;

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel.calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

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
    vector u_m = kernel.get_velocity_m(q);
    vector u_p = kernel.get_velocity_p(q);

    vector delta_u_m; // set exterior value to zero
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_p = -integrator_p.get_normal_vector(q);

    vector flux =
      kernel.calculate_flux_linearized_interior(u_p, u_m, delta_u_p, delta_u_m, normal_p);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  // make sure that this function is only accessed for OperatorType::homogeneous
  AssertThrow(
    operator_type == OperatorType::homogeneous,
    ExcMessage(
      "For the linearized convective operator, only OperatorType::homogeneous makes sense."));

  BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = kernel.get_velocity_m(q);
    vector u_p = kernel.calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

    vector delta_u_m = integrator.get_value(q);
    vector delta_u_p =
      kernel.calculate_exterior_value_linearized(delta_u_m, q, integrator, boundary_type);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel.calculate_flux_linearized_boundary(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace IncNS
