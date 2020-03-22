#include "laplace_operator.h"

#include "../../convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h"

namespace Poisson
{
template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit(
  MatrixFree<dim, Number> const &        matrix_free,
  AffineConstraints<double> const &      constraint_matrix,
  LaplaceOperatorData<rank, dim> const & data)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  kernel.reinit(matrix_free, data.kernel_data, data.dof_index);

  this->integrator_flags = kernel.get_integrator_flags(this->is_dg);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::calculate_penalty_parameter(
  MatrixFree<dim, Number> const & matrix_free,
  unsigned int const              dof_index)
{
  kernel.calculate_penalty_parameter(matrix_free, dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::update_after_mesh_movement()
{
  calculate_penalty_parameter(this->get_matrix_free(), this->get_data().dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::rhs_add_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & src) const
{
  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  this->matrix_free->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector,
                          this,
                          tmp,
                          src,
                          false /*zero_dst_vector = false*/);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel.reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel.reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face_cell_based(
  unsigned int const       cell,
  unsigned int const       face,
  types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(integrator.get_gradient(q), q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_integral(IntegratorFace & integrator_m,
                                                             IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m = integrator_m.get_value(q);
    value value_p = integrator_p.get_value(q);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = integrator_p.get_normal_derivative(q);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_p.submit_normal_derivative(gradient_flux, q);

    integrator_m.submit_value(-value_flux, q);
    integrator_p.submit_value(value_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set exterior value to zero
    value value_m = integrator_m.get_value(q);
    value value_p = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    // set exterior value to zero
    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    value value_p = integrator_p.get_value(q);
    value value_m = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_p, value_m);

    // minus sign to get the correct normal vector n⁺ = -n⁻
    value normal_gradient_p = -integrator_p.get_normal_derivative(q);
    value normal_gradient_m = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_p, normal_gradient_m, value_p, value_m);

    integrator_p.submit_normal_derivative(-gradient_flux, q); // opposite sign since n⁺ = -n⁻
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral(
  IntegratorFace &           integrator_m,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  ConvDiff::BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      ConvDiff::calculate_interior_value<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);
    value value_p =
      ConvDiff::calculate_exterior_value<dim, Number, n_components, rank>(value_m,
                                                                          q,
                                                                          integrator_m,
                                                                          operator_type,
                                                                          boundary_type,
                                                                          boundary_id,
                                                                          this->data.bc,
                                                                          this->time);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      ConvDiff::calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                                    integrator_m,
                                                                                    operator_type);
    value normal_gradient_p =
      ConvDiff::calculate_exterior_normal_gradient<dim, Number, n_components, rank>(
        normal_gradient_m,
        q,
        integrator_m,
        operator_type,
        boundary_type,
        boundary_id,
        this->data.bc,
        this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_continuous(
  IntegratorFace &           integrator_m,
  types::boundary_id const & boundary_id) const
{
  ConvDiff::BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value neumann_value = ConvDiff::calculate_neumann_value<dim, Number, n_components, rank>(
      q, integrator_m, boundary_type, boundary_id, this->data.bc, this->time);

    integrator_m.submit_value(-neumann_value, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::fill_dirichlet_values_continuous(
  std::map<types::global_dof_index, double> & boundary_values,
  double const                                time) const
{
  for(auto dbc : this->data.bc->dirichlet_bc)
  {
    dbc.second->set_time(time);
    VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                             this->matrix_free->get_dof_handler(
                                               this->data.dof_index),
                                             dbc.first,
                                             *dbc.second,
                                             boundary_values);
  }

  // TODO extend to dirichlet_mortar_bc
  AssertThrow(
    this->data.bc->dirichlet_mortar_bc.empty(),
    ExcMessage(
      "Dirichlet boundary conditions of mortar type are currently not implemented for continuous elements."));
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_dirichlet_bc_from_dof_vector(
  IntegratorFace &           integrator_m,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  ConvDiff::BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      ConvDiff::calculate_interior_value<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);

    // deviating from the standard boundary_face_loop_inhom_operator() function,
    // because the boundary condition comes from the vector src
    Assert(operator_type == OperatorType::inhomogeneous,
           ExcMessage("This function is only implemented for OperatorType::inhomogeneous."));

    value value_p = value();
    if(boundary_type == ConvDiff::BoundaryType::Dirichlet)
    {
      value_p = 2.0 * integrator_m.get_value(q);
    }
    else if(boundary_type == ConvDiff::BoundaryType::Neumann)
    {
      // do nothing
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      ConvDiff::calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                                    integrator_m,
                                                                                    operator_type);
    value normal_gradient_p =
      ConvDiff::calculate_exterior_normal_gradient<dim, Number, n_components, rank>(
        normal_gradient_m,
        q,
        integrator_m,
        operator_type,
        boundary_type,
        boundary_id,
        this->data.bc,
        this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::cell_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::face_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    MatrixFree<dim, Number> const & matrix_free,
    VectorType &                    dst,
    VectorType const &              src,
    Range const &                   range) const
{
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    // deviating from the standard function boundary_face_loop_inhom_operator()
    // because the boundary condition comes from the vector src
    this->integrator_m->gather_evaluate(src,
                                        this->integrator_flags.face_evaluate.value,
                                        this->integrator_flags.face_evaluate.gradient);

    do_boundary_integral_dirichlet_bc_from_dof_vector(*this->integrator_m,
                                                      OperatorType::inhomogeneous,
                                                      matrix_free.get_boundary_id(face));

    this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);
  }
}

template class LaplaceOperator<2, float, 1>;
template class LaplaceOperator<2, double, 1>;
template class LaplaceOperator<2, float, 2>;
template class LaplaceOperator<2, double, 2>;

template class LaplaceOperator<3, float, 1>;
template class LaplaceOperator<3, double, 1>;
template class LaplaceOperator<3, float, 3>;
template class LaplaceOperator<3, double, 3>;

} // namespace Poisson
