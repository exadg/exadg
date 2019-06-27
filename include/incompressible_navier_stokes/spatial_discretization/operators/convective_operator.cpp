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
LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel->get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ConvectiveOperatorData<dim> const & data)
{
  (void)matrix_free;
  (void)constraint_matrix;
  (void)data;

  AssertThrow(false,
              ExcMessage(
                "This reinit() function can not be used to initialize the viscous operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &                           matrix_free,
  AffineConstraints<double> const &                         constraint_matrix,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel)
{
  kernel = convective_kernel;

  Base::reinit(matrix_free, constraint_matrix, data);

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
                          MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values);
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
                          MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_linear_transport(
  VectorType &       dst,
  VectorType const & src,
  Number const       time,
  VectorType const & velocity_linear_transport) const
{
  this->set_velocity_linear_transport(velocity_linear_transport);

  this->time = time;

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
  IntegratorCell integrator(matrix_free, this->data.dof_index, this->data.quad_index_nonlinear);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator.gather_evaluate(src,
                               this->integrator_flags.cell_evaluate.value,
                               this->integrator_flags.cell_evaluate.gradient,
                               this->integrator_flags.cell_evaluate.hessian);

    do_cell_integral_nonlinear_operator(integrator);

    integrator.integrate_scatter(this->integrator_flags.cell_integrate.value,
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
  IntegratorFace integrator_m(matrix_free,
                              true,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);
  IntegratorFace integrator_p(matrix_free,
                              false,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src,
                                 this->integrator_flags.face_evaluate.value,
                                 this->integrator_flags.face_evaluate.gradient);

    integrator_p.gather_evaluate(src,
                                 this->integrator_flags.face_evaluate.value,
                                 this->integrator_flags.face_evaluate.gradient);

    do_face_integral_nonlinear_operator(integrator_m, integrator_p);

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate.value,
                                   this->integrator_flags.face_integrate.gradient,
                                   dst);

    integrator_p.integrate_scatter(this->integrator_flags.face_integrate.value,
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
  IntegratorFace integrator_m(matrix_free,
                              true,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src,
                                 this->integrator_flags.face_evaluate.value,
                                 this->integrator_flags.face_evaluate.gradient);

    do_boundary_integral_nonlinear_operator(integrator_m, matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate.value,
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

    if(this->data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F = outer_product(u, u);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(this->data.kernel_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: (u * grad) u = grad(u) * u
      tensor gradient_u = integrator.get_gradient(q);
      vector F          = gradient_u * u;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else if(this->data.kernel_data.formulation ==
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
      kernel->calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m);

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
  BoundaryTypeU boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = kernel->calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->data.bc, this->time);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel->calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
IntegratorFlags
ConvectiveOperator<dim, Number>::get_integrator_flags_linear_transport() const
{
  IntegratorFlags flags;

  if(this->data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
  {
    flags.cell_evaluate  = CellFlags(true, false, false);
    flags.cell_integrate = CellFlags(false, true, false);

    flags.face_evaluate  = FaceFlags(true, false);
    flags.face_integrate = FaceFlags(true, false);
  }
  else if(this->data.kernel_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
  {
    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(true, false, false);

    flags.face_evaluate  = FaceFlags(true, false);
    flags.face_integrate = FaceFlags(true, false);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return flags;
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_linear_transport(VectorType const & src) const
{
  velocity_linear_transport = &src;
  velocity_linear_transport->update_ghost_values();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_linear_transport(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   cell_range) const
{
  IntegratorCell integrator(matrix_free, this->data.dof_index, this->data.quad_index_nonlinear);

  IntegratorCell integrator_velocity(matrix_free,
                                     this->data.dof_index,
                                     this->data.quad_index_nonlinear);

  IntegratorFlags flags = get_integrator_flags_linear_transport();

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    integrator.gather_evaluate(src,
                               flags.cell_evaluate.value,
                               flags.cell_evaluate.gradient,
                               flags.cell_evaluate.hessian);

    // Note that the integrator flags which are valid here are different from those for the
    // linearized operator!
    integrator_velocity.reinit(cell);
    integrator_velocity.gather_evaluate(*velocity_linear_transport, true, false, false);

    do_cell_integral_linear_transport(integrator, integrator_velocity);

    integrator.integrate_scatter(flags.cell_integrate.value, flags.cell_integrate.gradient, dst);
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
  IntegratorFace integrator_m(matrix_free,
                              true,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);

  IntegratorFace integrator_p(matrix_free,
                              false,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);

  IntegratorFace integrator_velocity_m(matrix_free,
                                       true,
                                       this->data.dof_index,
                                       this->data.quad_index_nonlinear);

  IntegratorFace integrator_velocity_p(matrix_free,
                                       false,
                                       this->data.dof_index,
                                       this->data.quad_index_nonlinear);


  IntegratorFlags flags = get_integrator_flags_linear_transport();

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.gather_evaluate(src, flags.face_evaluate.value, flags.face_evaluate.gradient);

    integrator_p.gather_evaluate(src, flags.face_evaluate.value, flags.face_evaluate.gradient);

    integrator_velocity_m.reinit(face);
    integrator_velocity_m.gather_evaluate(*velocity_linear_transport, true, false);

    integrator_velocity_p.reinit(face);
    integrator_velocity_p.gather_evaluate(*velocity_linear_transport, true, false);

    do_face_integral_linear_transport(integrator_m,
                                      integrator_p,
                                      integrator_velocity_m,
                                      integrator_velocity_p);

    integrator_m.integrate_scatter(flags.face_integrate.value, flags.face_integrate.gradient, dst);

    integrator_p.integrate_scatter(flags.face_integrate.value, flags.face_integrate.gradient, dst);
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
  IntegratorFace integrator_m(matrix_free,
                              true,
                              this->data.dof_index,
                              this->data.quad_index_nonlinear);

  IntegratorFace integrator_velocity_m(matrix_free,
                                       true,
                                       this->data.dof_index,
                                       this->data.quad_index_nonlinear);

  IntegratorFlags flags = get_integrator_flags_linear_transport();

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_m.gather_evaluate(src, flags.face_evaluate.value, flags.face_evaluate.gradient);

    integrator_velocity_m.reinit(face);
    integrator_velocity_m.gather_evaluate(*velocity_linear_transport, true, false);

    do_boundary_integral_linear_transport(integrator_m,
                                          integrator_velocity_m,
                                          matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(flags.face_integrate.value, flags.face_integrate.gradient, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_linear_transport(IntegratorCell & integrator,
                                                                   IntegratorCell & velocity) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector w = velocity.get_value(q);

    if(this->data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F = uw
      vector u = integrator.get_value(q);
      tensor F = outer_product(u, w);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(this->data.kernel_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
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
  IntegratorFace & integrator_p,
  IntegratorFace & velocity_m,
  IntegratorFace & velocity_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = integrator_m.get_value(q);
    vector u_p = integrator_p.get_value(q);

    vector w_m = velocity_m.get_value(q);
    vector w_p = velocity_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux =
      kernel->calculate_flux_linear_transport_interior_and_neighbor(u_m, u_p, w_m, w_p, normal_m);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_linear_transport(
  IntegratorFace &           integrator,
  IntegratorFace &           velocity,
  types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = kernel->calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->data.bc, this->time);

    // concerning the transport velocity w, use the same value for interior and
    // exterior states, i.e., do not prescribe boundary conditions
    vector w_m = velocity.get_value(q);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linear_transport_boundary(u_m, u_p, w_m, w_m, normal_m, boundary_type);

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
ConvectiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
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
    vector u       = kernel->get_velocity_cell(q);

    if(this->data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      tensor flux = kernel->get_volume_flux_linearized_divergence_formulation(u, delta_u);

      integrator.submit_gradient(flux, q);
    }
    else if(this->data.kernel_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      tensor grad_u       = kernel->get_velocity_gradient_cell(q);
      tensor grad_delta_u = integrator.get_gradient(q);

      vector flux =
        kernel->get_volume_flux_linearized_convective_formulation(u, delta_u, grad_u, grad_delta_u);

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
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux = kernel->calculate_flux_linearized_interior_and_neighbor(
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
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

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
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m);

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
      kernel->calculate_flux_linearized_interior(u_p, u_m, delta_u_p, delta_u_m, normal_p);

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

  BoundaryTypeU boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->calculate_exterior_value_nonlinear(
      u_m, q, integrator, boundary_type, boundary_id, this->data.bc, this->time);

    vector delta_u_m = integrator.get_value(q);
    vector delta_u_p =
      kernel->calculate_exterior_value_linearized(delta_u_m, q, integrator, boundary_type);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel->calculate_flux_linearized_boundary(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace IncNS
