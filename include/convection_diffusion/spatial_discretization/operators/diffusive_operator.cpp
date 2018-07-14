#include "diffusive_operator.h"

namespace ConvDiff {

template <int dim, int fe_degree, typename value_type>
void DiffusiveOperator<dim, fe_degree, value_type>::do_cell_integral(
    FEEvalCell &fe_eval) const {

  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_gradient(fe_eval.get_gradient(q)*diffusivity,q);
  
}

template <int dim, int fe_degree, typename value_type>
void DiffusiveOperator<dim, fe_degree, value_type>::do_face_integral(
    FEEvalFace &fe_eval, FEEvalFace &fe_eval_neighbor) const {
  VectorizedArray<value_type> tau_IP =
      std::max(fe_eval.read_cell_data(array_penalty_parameter),
               fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
      IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
    VectorizedArray<value_type> jump_value =
        fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    VectorizedArray<value_type> normal_gradient_m =
        fe_eval.get_normal_gradient(q);
    VectorizedArray<value_type> normal_gradient_p =
        fe_eval_neighbor.get_normal_gradient(q);
    VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(
        normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval_neighbor.submit_normal_gradient(value_flux, q);

    fe_eval.submit_value(-gradient_flux, q);
    fe_eval_neighbor.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
  }

}

template <int dim, int fe_degree, typename value_type>
void DiffusiveOperator<dim, fe_degree, value_type>::do_face_int_integral(
    FEEvalFace &fe_eval, FEEvalFace &fe_eval_neighbor) const {
  VectorizedArray<value_type> tau_IP =
      std::max(fe_eval.read_cell_data(array_penalty_parameter),
               fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
      IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
    // set exterior value to zero
    VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    // set exterior value to zero
    VectorizedArray<value_type> normal_gradient_m =
        fe_eval.get_normal_gradient(q);
    VectorizedArray<value_type> normal_gradient_p =
        make_vectorized_array<value_type>(0.0);
    VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(
        normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }

}

template <int dim, int fe_degree, typename value_type>
void DiffusiveOperator<dim, fe_degree, value_type>::do_face_ext_integral(
    FEEvalFace &fe_eval, FEEvalFace &fe_eval_neighbor) const {
  VectorizedArray<value_type> tau_IP =
      std::max(fe_eval.read_cell_data(array_penalty_parameter),
               fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
      IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
    // set value_m to zero
    VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    // set gradient_m to zero
    VectorizedArray<value_type> normal_gradient_m =
        make_vectorized_array<value_type>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    VectorizedArray<value_type> normal_gradient_p =
        -fe_eval_neighbor.get_normal_gradient(q);
    VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(
        normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval_neighbor.submit_normal_gradient(-value_flux,
                                            q); // minus sign since n⁺ = -n⁻
    fe_eval_neighbor.submit_value(-gradient_flux, q);
  }

}

template <int dim, int fe_degree, typename value_type>
void DiffusiveOperator<dim, fe_degree, value_type>::do_boundary_integral(
    FEEvalFace &fe_eval, OperatorType const &operator_type,
    types::boundary_id const &boundary_id) const {
  BoundaryType boundary_type = this->ad.get_boundary_type(boundary_id);

  VectorizedArray<value_type> tau_IP =
      fe_eval.read_cell_data(array_penalty_parameter) *
      IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
    VectorizedArray<value_type> value_m =
        calculate_interior_value(q, fe_eval, operator_type);
    VectorizedArray<value_type> value_p = calculate_exterior_value(
        value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    VectorizedArray<value_type> jump_value = value_m - value_p;
    VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

    VectorizedArray<value_type> normal_gradient_m =
        calculate_interior_normal_gradient(q, fe_eval, operator_type);
    VectorizedArray<value_type> normal_gradient_p =
        calculate_exterior_normal_gradient(normal_gradient_m, q, fe_eval,
                                           operator_type, boundary_type,
                                           boundary_id);
    VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(
        normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_gradient(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }

}
}
