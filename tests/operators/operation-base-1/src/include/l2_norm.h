#ifndef OPERATOR_BASE_L2_NORM
#define OPERATOR_BASE_L2_NORM


template <int dim, int fe_degree, typename value_type> class L2Norm {
public:
  typedef L2Norm<dim, fe_degree, value_type> This;

  L2Norm(MatrixFree<dim, value_type> const &mf_data) : data(&mf_data) {}

  double run(parallel::distributed::Vector<value_type> &src) const {
    tt = 0.0;
    data->cell_loop(&This::cell_loop, this, src, src);
    double temp = 0;
    for (unsigned int i = 0; i < VectorizedArray<double>::n_array_elements; i++)
      temp += tt[i];
    
    double local = temp;
    
    MPI_Reduce(&local, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    return std::sqrt(temp);
  }

private:
  template <typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const {
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
      VectorizedArray<value_type> rhs = make_vectorized_array<value_type>(0.0);
      rhs = 1.0;
      fe_eval.submit_value(rhs, q);
    }
    fe_eval.integrate(true, false);
  }

  void
  cell_loop(MatrixFree<dim, value_type> const &data,
            parallel::distributed::Vector<value_type> &,
            parallel::distributed::Vector<value_type> const &src,
            std::pair<unsigned int, unsigned int> const &cell_range) const {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, value_type> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, true, false, false);
      VectorizedArray<double> ttt;
      ttt = 0;
      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
        ttt += fe_eval.begin_values()[q] * fe_eval.begin_values()[q] *
              fe_eval.JxW(q);
      }
      for(unsigned int i = 0; i < data.n_active_entries_per_cell_batch(cell); i++)
          tt[i]+=ttt[i];
    }
  }
  mutable VectorizedArray<double> tt;
  MatrixFree<dim, value_type> const *data;
};

#endif