namespace Laplace
{
template<int dim, int fe_degree, typename value_type>
RHSOperator<dim, fe_degree, value_type>::RHSOperator() : data(nullptr), eval_time(0.0)
{
}

template<int dim, int fe_degree, typename value_type>
void
RHSOperator<dim, fe_degree, value_type>::initialize(MF const &             mf_data,
                                                    AdditionalData const & operator_data_in)
{
  this->data          = &mf_data;
  this->operator_data = operator_data_in;
}

template<int dim, int fe_degree, typename value_type>
void
RHSOperator<dim, fe_degree, value_type>::evaluate(VectorType & dst, value_type const evaluation_time) const
{
  dst = 0;
  evaluate_add(dst, evaluation_time);
}

template<int dim, int fe_degree, typename value_type>
void
RHSOperator<dim, fe_degree, value_type>::evaluate_add(VectorType &     dst,
                                                      value_type const evaluation_time) const
{
  this->eval_time = evaluation_time;
  data->cell_loop(&This::cell_loop, this, dst, dst);
}

template<int dim, int fe_degree, typename value_type>
void
RHSOperator<dim, fe_degree, value_type>::cell_loop(MF const &   data,
                                                   VectorType & dst,
                                                   VectorType const &,
                                                   Range const & cell_range) const
{
  FECellEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      auto q_points = fe_eval.quadrature_point(q);
      auto rhs      = make_vectorized_array<value_type>(0.0);
      evaluate_scalar_function(rhs, operator_data.rhs, q_points, eval_time);
      fe_eval.submit_value(rhs, q);
    }

    fe_eval.integrate_scatter(true, false, dst);
  }
}
} // namespace Laplace