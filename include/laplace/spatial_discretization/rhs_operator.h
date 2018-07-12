#ifndef LAPLACE_RHS
#define LAPLACE_RHS

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../include/functionalities/evaluate_functions.h"

namespace Laplace {

template <int dim> struct RHSOperatorData {
  RHSOperatorData() : dof_index(0), quad_index(0) {}

  unsigned int dof_index;
  unsigned int quad_index;
  std::shared_ptr<Function<dim>> rhs;
};

template <int dim, int fe_degree, typename value_type> class RHSOperator {
public:
  typedef RHSOperator<dim, fe_degree, value_type> This;
  typedef FEEvaluation<dim, fe_degree, fe_degree + 1, 1, value_type> FECellEval;
  typedef parallel::distributed::Vector<value_type> VNumber;
  typedef std::pair<unsigned int, unsigned int> Range;
  typedef MatrixFree<dim, value_type> MF;
  typedef RHSOperatorData<dim> AdditionalData;

  RHSOperator() : data(nullptr), eval_time(0.0) {}

  void initialize(MF const &mf_data, AdditionalData const &operator_data_in) {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void evaluate(VNumber &dst, value_type const evaluation_time) const {
    dst = 0;
    evaluate_add(dst, evaluation_time);
  }

  void evaluate_add(VNumber &dst, value_type const evaluation_time) const {
    this->eval_time = evaluation_time;
    data->cell_loop(&This::cell_loop, this, dst, dst);
  }

private:
  void cell_loop(MF const &data, VNumber &dst, VNumber const &,
                 Range const &cell_range) const {
    FECellEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for (auto cell = cell_range.first; cell < cell_range.second; ++cell) {
      fe_eval.reinit(cell);

      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
        auto q_points = fe_eval.quadrature_point(q);
        auto rhs = make_vectorized_array<value_type>(0.0);
        evaluate_scalar_function(rhs, operator_data.rhs, q_points, eval_time);
        fe_eval.submit_value(rhs, q);
      }

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  MF const *data;
  RHSOperatorData<dim> operator_data;
  value_type mutable eval_time;
};
}

#endif