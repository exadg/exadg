#include "laplace_operator.h"

namespace Laplace
{

template <int dim, int degree, typename Number>
LaplaceOperator<dim, degree, Number>::LaplaceOperator()
    : OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>>() {}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim, degree, Number>::do_cell_integral(
    FEEvalCell &p) const {
  for (unsigned int q = 0; q < p.n_q_points; ++q)
    p.submit_gradient(p.get_gradient(q), q);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim, degree, Number>::do_face_integral(
    FEEvalFace &p_n, FEEvalFace &p_p) const {
  VectorizedArray<Number> sigmaF =
        std::max(p_n.read_cell_data(array_penalty_parameter),
                 p_p.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < p_n.n_q_points; ++q) {
    VectorizedArray<Number> valueM = p_n.get_value(q);
    VectorizedArray<Number> valueP = p_p.get_value(q);

    VectorizedArray<Number> jump_value = valueM - valueP;
    VectorizedArray<Number> average_gradient =
        (p_n.get_normal_gradient(q) + p_p.get_normal_gradient(q)) * 0.5;
    average_gradient = average_gradient - jump_value * sigmaF;

    p_n.submit_normal_gradient(-0.5 * jump_value, q);
    p_p.submit_normal_gradient(-0.5 * jump_value, q);
    p_n.submit_value(-average_gradient, q);
    p_p.submit_value(average_gradient, q);
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim, degree, Number>::do_face_int_integral(
    FEEvalFace &p_n, FEEvalFace &p_p) const {
  VectorizedArray<Number> sigmaF =
        std::max(p_n.read_cell_data(array_penalty_parameter),
                 p_p.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < p_n.n_q_points; ++q) {
    VectorizedArray<Number> valueM = p_n.get_value(q);

    VectorizedArray<Number> jump_value = valueM;
    VectorizedArray<Number> average_gradient =
        (p_n.get_normal_gradient(q)) * 0.5;
    average_gradient = average_gradient - jump_value * sigmaF;

    p_n.submit_normal_gradient(-0.5 * jump_value, q);
    p_n.submit_value(-average_gradient, q);
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim, degree, Number>::do_face_ext_integral(
    FEEvalFace &p_n, FEEvalFace &p_p) const {
  VectorizedArray<Number> sigmaF =
        std::max(p_n.read_cell_data(array_penalty_parameter),
                 p_p.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(degree, this->ad.IP_factor);

  for (unsigned int q = 0; q < p_p.n_q_points; ++q) {
    VectorizedArray<Number> valueP = p_p.get_value(q);

    VectorizedArray<Number> jump_value = -valueP;
    VectorizedArray<Number> average_gradient =
        (p_p.get_normal_gradient(q)) * 0.5;
    average_gradient = average_gradient - jump_value * sigmaF;

    p_p.submit_normal_gradient(-0.5 * jump_value, q);
    p_p.submit_value(average_gradient, q);
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim, degree, Number>::do_boundary_integral(
    FEEvalFace & phi, OperatorType const & /**/,
    types::boundary_id const & bid) const { 
  VectorizedArray<Number> sigmaF =
        phi.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<Number>(degree, this->ad.IP_factor);
  const auto bt = this->ad.get_boundary_type(bid);
    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      if (bt == BoundaryType::dirichlet) {
        VectorizedArray<Number> valueM = phi.get_value(q);

        VectorizedArray<Number> jump_value = 2.0 * valueM;
        VectorizedArray<Number> average_gradient =
            phi.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

        phi.submit_normal_gradient(-0.5 * jump_value, q);
        phi.submit_value(-average_gradient, q);
      } else if (bt == BoundaryType::neumann) {
        VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
        VectorizedArray<Number> average_gradient =
            make_vectorized_array<Number>(0.0);
        average_gradient = average_gradient - jump_value * sigmaF;

        phi.submit_normal_gradient(-0.5 * jump_value, q);
        phi.submit_value(-average_gradient, q);
      }
    }
}

}