#include "laplace_operator.h"

#include "weak_boundary_conditions.h"

namespace Poisson
{
namespace Operators
{
template<int dim, typename Number>
LaplaceKernel<dim, Number>::LaplaceKernel() : tau(make_vectorized_array<Number>(0.0))
{
}

template<int dim, typename Number>
void
LaplaceKernel<dim, Number>::reinit(MatrixFree<dim, Number> const & matrix_free,
                                   LaplaceKernelData const &       data_in,
                                   unsigned int const              dof_index) const
{
  data = data_in;

  MappingQGeneric<dim> mapping(data_in.degree_mapping);
  IP::calculate_penalty_parameter<dim, Number>(
    array_penalty_parameter, matrix_free, mapping, data_in.degree, dof_index);
}

template<int dim, typename Number>
void
LaplaceKernel<dim, Number>::reinit_face(IntegratorFace & integrator_m,
                                        IntegratorFace & integrator_p) const
{
  tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                 integrator_p.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
}

template<int dim, typename Number>
void
LaplaceKernel<dim, Number>::reinit_boundary_face(IntegratorFace & integrator_m) const
{
  tau = integrator_m.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
}

template<int dim, typename Number>
void
LaplaceKernel<dim, Number>::reinit_face_cell_based(types::boundary_id const boundary_id,
                                                   IntegratorFace &         integrator_m,
                                                   IntegratorFace &         integrator_p) const
{
  if(boundary_id == numbers::internal_face_boundary_id) // internal face
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
  }
  else // boundary face
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
  }
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceKernel<dim, Number>::calculate_value_flux(scalar const & value_m,
                                                   scalar const & value_p) const
{
  return -0.5 * (value_m - value_p);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceKernel<dim, Number>::calculate_gradient_flux(scalar const & normal_gradient_m,
                                                      scalar const & normal_gradient_p,
                                                      scalar const & value_m,
                                                      scalar const & value_p) const
{
  return 0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  LaplaceKernel<dim, Number>::get_volume_flux(IntegratorCell &   integrator,
                                              unsigned int const q) const
{
  return integrator.get_gradient(q);
}

template class LaplaceKernel<2, float>;
template class LaplaceKernel<2, double>;

template class LaplaceKernel<3, float>;
template class LaplaceKernel<3, double>;

} // namespace Operators

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free,
                                     AffineConstraints<double> const & constraint_matrix,
                                     LaplaceOperatorData<dim> const &  operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  kernel.reinit(matrix_free, operator_data.kernel_data, operator_data.dof_index);
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel.reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel.reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                     unsigned int const       face,
                                                     types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(kernel.get_volume_flux(integrator, q), q);
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                               IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    scalar value_flux = kernel.calculate_value_flux(value_m, value_p);

    scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
    scalar normal_gradient_p = integrator_p.get_normal_derivative(q);

    scalar gradient_flux =
      kernel.calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(value_flux, q);
    integrator_p.submit_normal_derivative(value_flux, q);

    integrator_m.submit_value(-gradient_flux, q);
    integrator_p.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                   IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set exterior value to zero
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = make_vectorized_array<Number>(0.0);

    scalar value_flux = kernel.calculate_value_flux(value_m, value_p);

    // set exterior value to zero
    scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
    scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

    scalar gradient_flux =
      kernel.calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(value_flux, q);
    integrator_m.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                   IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_p = integrator_p.get_value(q);
    scalar value_m = make_vectorized_array<Number>(0.0);

    scalar value_flux = kernel.calculate_value_flux(value_p, value_m);

    // set gradient_m to zero
    scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    scalar normal_gradient_p = -integrator_p.get_normal_derivative(q);

    scalar gradient_flux =
      kernel.calculate_gradient_flux(normal_gradient_p, normal_gradient_m, value_p, value_m);

    integrator_p.submit_normal_derivative(-value_flux, q); // opposite sign since n⁺ = -n⁻
    integrator_p.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator_m,
                                                   OperatorType const &       operator_type,
                                                   types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, integrator_m, operator_type);
    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->operator_data.bc,
                                              this->eval_time);

    scalar value_flux = kernel.calculate_value_flux(value_m, value_p);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, integrator_m, operator_type);
    scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                  q,
                                                                  integrator_m,
                                                                  operator_type,
                                                                  boundary_type,
                                                                  boundary_id,
                                                                  this->operator_data.bc,
                                                                  this->eval_time);

    scalar gradient_flux =
      kernel.calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(value_flux, q);
    integrator_m.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  LaplaceOperatorData<dim> const &     operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  unsigned int counter = 0;
  if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
    counter++;

  if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
    counter++;

  if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
    counter++;

  AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
}

template class LaplaceOperator<2, float>;
template class LaplaceOperator<2, double>;

template class LaplaceOperator<3, float>;
template class LaplaceOperator<3, double>;

} // namespace Poisson
