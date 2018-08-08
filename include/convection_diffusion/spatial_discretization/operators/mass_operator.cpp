#include "mass_operator.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
void
MassMatrixOperator<dim, fe_degree, value_type>::initialize(
  MatrixFree<dim, value_type> const & mf_data,
  MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in)
{
  ConstraintMatrix constraint_matrix;
  Parent::reinit(mf_data, constraint_matrix, mass_matrix_operator_data_in);
}

template<int dim, int fe_degree, typename value_type>
void
MassMatrixOperator<dim, fe_degree, value_type>::do_cell_integral(FEEvalCell & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_value(fe_eval.get_value(q), q);
}
} // namespace ConvDiff

#include "mass_operator.hpp"