#include "diffusive_operator.h"

#include "verify_boundary_conditions.h"
#include "weak_boundary_conditions.h"

#include "../../../operators/interior_penalty_parameter.h"

namespace ConvDiff
{
template<int dim, typename Number>
DiffusiveOperator<dim, Number>::DiffusiveOperator()
  : diffusivity(-1.0), tau(make_vectorized_array<Number>(0.0))
{
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &    matrix_free,
                                       AffineConstraints<double> const &  constraint_matrix,
                                       DiffusiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  MappingQGeneric<dim> mapping(operator_data.degree_mapping);
  IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                               *this->matrix_free,
                                               mapping,
                                               this->operator_data.degree,
                                               this->operator_data.dof_index);

  diffusivity = this->operator_data.diffusivity;

  AssertThrow(diffusivity > 0.0, ExcMessage("Diffusivity is not set!"));
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_value_flux(scalar const & value_m,
                                                       scalar const & value_p) const
{
  return -0.5 * diffusivity * (value_m - value_p);
}


/*
 *  Calculation of gradient flux. Strictly speaking, this value is not a
 * numerical flux since
 *  the flux is multiplied by the normal vector, i.e., "gradient_flux" =
 * numerical_flux * normal,
 *  where normal denotes the normal vector of element e⁻.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_gradient_flux(scalar const & normal_gradient_m,
                                                          scalar const & normal_gradient_p,
                                                          scalar const & value_m,
                                                          scalar const & value_p,
                                                          scalar const & penalty_parameter) const
{
  return diffusivity * 0.5 * (normal_gradient_m + normal_gradient_p) -
         diffusivity * penalty_parameter * (value_m - value_p);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  DiffusiveOperator<dim, Number>::get_volume_flux(FEEvalCell & fe_eval, unsigned int const q) const
{
  return fe_eval.get_gradient(q) * diffusivity;
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  tau = std::max(this->fe_eval_m->read_cell_data(array_penalty_parameter),
                 this->fe_eval_p->read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  tau = this->fe_eval_m->read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                       unsigned int const       face,
                                                       types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(boundary_id == numbers::internal_face_boundary_id) // internal face
  {
    tau = std::max(this->fe_eval_m->read_cell_data(array_penalty_parameter),
                   this->fe_eval_p->read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);
  }
  else // boundary face
  {
    tau = this->fe_eval_m->read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_cell_integral(FEEvalCell & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    fe_eval.submit_gradient(get_volume_flux(fe_eval, q), q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_integral(FEEvalFace & fe_eval_m,
                                                 FEEvalFace & fe_eval_p) const
{
  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    scalar value_m = fe_eval_m.get_value(q);
    scalar value_p = fe_eval_p.get_value(q);

    scalar value_flux = calculate_value_flux(value_m, value_p);

    scalar normal_gradient_m = fe_eval_m.get_normal_derivative(q);
    scalar normal_gradient_p = fe_eval_p.get_normal_derivative(q);

    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p, tau);

    fe_eval_m.submit_normal_derivative(value_flux, q);
    fe_eval_p.submit_normal_derivative(value_flux, q);

    fe_eval_m.submit_value(-gradient_flux, q);
    fe_eval_p.submit_value(gradient_flux, q); // opposite sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_int_integral(FEEvalFace & fe_eval_m,
                                                     FEEvalFace & fe_eval_p) const
{
  (void)fe_eval_p;

  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    // set exterior value to zero
    scalar value_m = fe_eval_m.get_value(q);
    scalar value_p = make_vectorized_array<Number>(0.0);

    scalar value_flux = calculate_value_flux(value_m, value_p);

    // set exterior value to zero
    scalar normal_gradient_m = fe_eval_m.get_normal_derivative(q);
    scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p, tau);

    fe_eval_m.submit_normal_derivative(value_flux, q);
    fe_eval_m.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_ext_integral(FEEvalFace & fe_eval_m,
                                                     FEEvalFace & fe_eval_p) const
{
  (void)fe_eval_m;

  for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = make_vectorized_array<Number>(0.0);
    scalar value_p = fe_eval_p.get_value(q);

    scalar value_flux = calculate_value_flux(value_p, value_m);

    // set gradient_m to zero
    scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    scalar normal_gradient_p = -fe_eval_p.get_normal_derivative(q);

    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_p, normal_gradient_m, value_p, value_m, tau);

    fe_eval_p.submit_normal_derivative(-value_flux, q); // opposite sign since n⁺ = -n⁻
    fe_eval_p.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_boundary_integral(FEEvalFace &               fe_eval_m,
                                                     OperatorType const &       operator_type,
                                                     types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, fe_eval_m, operator_type);

    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              fe_eval_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->operator_data.bc,
                                              this->eval_time);

    scalar value_flux = calculate_value_flux(value_m, value_p);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval_m, operator_type);

    scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                  q,
                                                                  fe_eval_m,
                                                                  operator_type,
                                                                  boundary_type,
                                                                  boundary_id,
                                                                  this->operator_data.bc,
                                                                  this->eval_time);

    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, value_m, value_p, tau);

    fe_eval_m.submit_normal_derivative(value_flux, q);
    fe_eval_m.submit_value(-gradient_flux, q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  DiffusiveOperatorData<dim> const &   operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

template class DiffusiveOperator<2, float>;
template class DiffusiveOperator<2, double>;

template class DiffusiveOperator<3, float>;
template class DiffusiveOperator<3, double>;

} // namespace ConvDiff
