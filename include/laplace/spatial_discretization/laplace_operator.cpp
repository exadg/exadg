#include "laplace_operator.h"

#include <navier_constants.h>
#include "../../functionalities/evaluate_functions.h"

namespace Laplace
{
template<int dim, int degree, typename Number>
LaplaceOperator<dim, degree, Number>::LaplaceOperator()
  : OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>>()
{
}

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_value_flux(
  VectorizedArray<Number> const & jump_value) const
{
  return -0.5 * /*diffusivity * */ jump_value;
}

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_interior_value(unsigned int const   q,
                                                                  FEEvalFace const &   fe_eval,
                                                                  OperatorType const & operator_type) const
{
  VectorizedArray<Number> value_m = make_vectorized_array<Number>(0.0);

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

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_exterior_value(VectorizedArray<Number> const & value_m,
                                                                  unsigned int const              q,
                                                                  FEEvalFace const &              fe_eval,
                                                                  OperatorType const &     operator_type,
                                                                  BoundaryType const &     boundary_type,
                                                                  types::boundary_id const boundary_id) const
{
  VectorizedArray<Number> value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      VectorizedArray<Number> g = make_vectorized_array<Number>(0.0);
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                                           = this->ad.bc->dirichlet_bc.find(boundary_id);
      Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);
      evaluate_scalar_function(g, it->second, q_points, this->eval_time);

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

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_gradient_flux(
  VectorizedArray<Number> const & normal_gradient_m,
  VectorizedArray<Number> const & normal_gradient_p,
  VectorizedArray<Number> const & jump_value,
  VectorizedArray<Number> const & penalty_parameter) const
{
  return /*diffusivity * */ 0.5 * (normal_gradient_m + normal_gradient_p) -
         /*diffusivity * */ penalty_parameter * jump_value;
}

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_interior_normal_gradient(
  unsigned int const   q,
  FEEvalFace const &   fe_eval,
  OperatorType const & operator_type) const
{
  VectorizedArray<Number> normal_gradient_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = fe_eval.get_normal_gradient(q);
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

template<int dim, int fe_degree, typename Number>
inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number>
                             LaplaceOperator<dim, fe_degree, Number>::calculate_exterior_normal_gradient(
  VectorizedArray<Number> const & normal_gradient_m,
  unsigned int const              q,
  FEEvalFace const &              fe_eval,
  OperatorType const &            operator_type,
  BoundaryType const &            boundary_type,
  types::boundary_id const        boundary_id) const
{
  VectorizedArray<Number> normal_gradient_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      VectorizedArray<Number> h = make_vectorized_array<Number>(0.0);
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                                           = this->ad.bc->neumann_bc.find(boundary_id);
      Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);
      evaluate_scalar_function(h, it->second, q_points, this->eval_time);

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

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & p) const
{
  for(unsigned int q = 0; q < p.n_q_points; ++q)
    p.submit_gradient(p.get_gradient(q), q);
}

template<int dim, int fe_degree, typename value_type>
void
LaplaceOperator<dim, fe_degree, value_type>::do_face_integral(FEEvalFace & fe_eval,
                                                              FEEvalFace & fe_eval_neighbor) const
{
  VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                                       IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);
#ifdef LAPALCE_CELL_TEST
  tau_IP = 100.0;
#endif
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    VectorizedArray<value_type> jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    VectorizedArray<value_type> normal_gradient_m = fe_eval.get_normal_gradient(q);
    VectorizedArray<value_type> normal_gradient_p = fe_eval_neighbor.get_normal_gradient(q);
    VectorizedArray<value_type> gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval_neighbor.submit_normal_gradient(value_flux, q);

    fe_eval.submit_value(-gradient_flux, q);
    fe_eval_neighbor.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, int fe_degree, typename value_type>
void
LaplaceOperator<dim, fe_degree, value_type>::do_face_int_integral(FEEvalFace & fe_eval,
                                                                  FEEvalFace & fe_eval_neighbor) const
{
  VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                                       IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);
#ifdef LAPALCE_CELL_TEST
  tau_IP = 100.0;
#endif

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set exterior value to zero
    VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    // set exterior value to zero
    VectorizedArray<value_type> normal_gradient_m = fe_eval.get_normal_gradient(q);
    VectorizedArray<value_type> normal_gradient_p = make_vectorized_array<value_type>(0.0);
    VectorizedArray<value_type> gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int fe_degree, typename value_type>
void
LaplaceOperator<dim, fe_degree, value_type>::do_face_ext_integral(FEEvalFace & fe_eval,
                                                                  FEEvalFace & fe_eval_neighbor) const
{
  VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                                       IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);
#ifdef LAPALCE_CELL_TEST
  tau_IP = 100.0;
#endif
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set value_m to zero
    VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    // set gradient_m to zero
    VectorizedArray<value_type> normal_gradient_m = make_vectorized_array<value_type>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    VectorizedArray<value_type> normal_gradient_p = -fe_eval_neighbor.get_normal_gradient(q);
    VectorizedArray<value_type> gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval_neighbor.submit_normal_gradient(-value_flux,
                                            q); // minus sign since n⁺ = -n⁻
    fe_eval_neighbor.submit_value(-gradient_flux, q);
  }
}

template<int dim, int fe_degree, typename value_type>
void
LaplaceOperator<dim, fe_degree, value_type>::do_boundary_integral(
  FEEvalFace &               fe_eval,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->ad.get_boundary_type(boundary_id);

  VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter) *
                                       IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);
#ifdef LAPALCE_CELL_TEST
  tau_IP = 100.0;
#endif
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    VectorizedArray<value_type> value_m = calculate_interior_value(q, fe_eval, operator_type);
    VectorizedArray<value_type> value_p =
      calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    VectorizedArray<value_type> jump_value = value_m - value_p;
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    VectorizedArray<value_type> normal_gradient_m =
      calculate_interior_normal_gradient(q, fe_eval, operator_type);
    VectorizedArray<value_type> normal_gradient_p = calculate_exterior_normal_gradient(
      normal_gradient_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    VectorizedArray<value_type> gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int fe_degree, typename Number>
MatrixOperatorBaseNew<dim, Number> *
LaplaceOperator<dim, fe_degree, Number>::get_new(unsigned int deg) const
{
  switch(deg)
  {
#if DEGREE_1
    case 1:
      return new LaplaceOperator<dim, 1, Number>();
#endif
#if DEGREE_2
    case 2:
      return new LaplaceOperator<dim, 2, Number>();
#endif
#if DEGREE_3
    case 3:
      return new LaplaceOperator<dim, 3, Number>();
#endif
#if DEGREE_4
    case 4:
      return new LaplaceOperator<dim, 4, Number>();
#endif
#if DEGREE_5
    case 5:
      return new LaplaceOperator<dim, 5, Number>();
#endif
#if DEGREE_6
    case 6:
      return new LaplaceOperator<dim, 6, Number>();
#endif
#if DEGREE_7
    case 7:
      return new LaplaceOperator<dim, 7, Number>();
#endif
#if DEGREE_8
    case 8:
      return new LaplaceOperator<dim, 8, Number>();
#endif
#if DEGREE_9
    case 9:
      return new LaplaceOperator<dim, 9, Number>();
#endif
    default:
      AssertThrow(false, ExcMessage("LaplaceOperator not implemented for this degree!"));
      return new LaplaceOperator<dim, 1, Number>(); // dummy return (statement not
                                                    // reached)
  }
}

} // namespace Laplace

#include "laplace_operator.hpp"