#include "diffusive_operator.h"

#include "verify_boundary_conditions.h"

#include "../../../functionalities/evaluate_functions.h"
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
DiffusiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &    mf_data,
                                       AffineConstraints<double> const &  constraint_matrix,
                                       DiffusiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);

  MappingQGeneric<dim> mapping(operator_data.degree_mapping);
  IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                               *this->matrix_free,
                                               mapping,
                                               this->operator_data.degree,
                                               this->operator_data.dof_index);

  diffusivity = this->operator_data.diffusivity;
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  AssertThrow(diffusivity > 0.0, ExcMessage("Diffusivity is not set!"));
  Base::apply_add(dst, src);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::apply_add(VectorType & /*dst*/,
                                          VectorType const & /*src*/,
                                          Number const /*time*/) const
{
  // This function has to be overwritten explicitly. Otherwise the compiler
  // complains that this function of the base class is hidden by the other apply_add
  AssertThrow(false,
              ExcMessage("DiffusiveOperator cannot be called with additional parameter time!"));
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
 *  The following two functions calculate the interior_value/exterior_value
 *  depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *                            +-----------------------------+-----------------------+
 *                            | Dirichlet boundaries        | Neumann boundaries    |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | full operator           | phi⁺ = -phi⁻ + 2g           | phi⁺ = phi⁻           |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | homogeneous operator    | phi⁺ = -phi⁻                | phi⁺ = phi⁻           |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | inhomogeneous operator  | phi⁺ = -phi⁻ + 2g, phi⁻ = 0 | phi⁺ = phi⁻, phi⁻ = 0 |
 *  +-------------------------+-----------------------------+-----------------------+
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_interior_value(unsigned int const   q,
                                                           FEEvalFace const &   fe_eval,
                                                           OperatorType const & operator_type) const
{
  scalar value_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    value_m = fe_eval.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    value_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_exterior_value(
    scalar const &           value_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar g = evaluate_scalar_function(it->second, q_points, this->eval_time);

      value_p = -value_m + 2.0 * g;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
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

// clang-format off
  /*
   *  The following two functions calculate the interior/exterior velocity gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n + 2h                    |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n                         |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁺)*n = grad(phi⁻)*n, grad(phi⁻)*n = 0 | grad(phi⁺)*n = -grad(phi⁻)*n + 2h, grad(phi⁻)*n  = 0 |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = 0                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                           | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   */
// clang-format on
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_interior_normal_gradient(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  scalar normal_gradient_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = fe_eval.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    normal_gradient_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, Number>::calculate_exterior_normal_gradient(
    scalar const &           normal_gradient_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->neumann_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar h = evaluate_scalar_function(it->second, q_points, this->eval_time);

      normal_gradient_p = -normal_gradient_m + 2.0 * h;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
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
    scalar value_p =
      calculate_exterior_value(value_m, q, fe_eval_m, operator_type, boundary_type, boundary_id);

    scalar value_flux = calculate_value_flux(value_m, value_p);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval_m, operator_type);
    scalar normal_gradient_p = calculate_exterior_normal_gradient(
      normal_gradient_m, q, fe_eval_m, operator_type, boundary_type, boundary_id);

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
  ConvDiff::do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

template class DiffusiveOperator<2, float>;
template class DiffusiveOperator<2, double>;

template class DiffusiveOperator<3, float>;
template class DiffusiveOperator<3, double>;

} // namespace ConvDiff
