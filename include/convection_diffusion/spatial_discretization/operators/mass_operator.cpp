#include "mass_operator.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
void
MassMatrixOperator<dim, fe_degree, value_type>::initialize(
  MatrixFree<dim, value_type> const & mf_data,
  MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in,
  unsigned int           level_mg_handler)
{
  ConstraintMatrix constraint_matrix;
  Parent::reinit(mf_data, constraint_matrix, mass_matrix_operator_data_in, level_mg_handler);
}

template<int dim, int fe_degree, typename value_type>
void
MassMatrixOperator<dim, fe_degree, value_type>::initialize(
  MatrixFree<dim, value_type> const & mf_data,
 ConstraintMatrix const& constraint_matrix,
  MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in,
  unsigned int           level_mg_handler)
{
  Parent::reinit(mf_data, constraint_matrix, mass_matrix_operator_data_in, level_mg_handler);
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