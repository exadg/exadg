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

  RHSOperator();

  void initialize(MF const &mf_data, AdditionalData const &operator_data_in);

  void evaluate(VNumber &dst, value_type const evaluation_time) const;

  void evaluate_add(VNumber &dst, value_type const evaluation_time) const;

private:
  void cell_loop(MF const &data, VNumber &dst, VNumber const &,
                 Range const &cell_range) const;

  MF const *data;
  RHSOperatorData<dim> operator_data;
  value_type mutable eval_time;
};
}

#include "rhs_operator.cpp"

#endif