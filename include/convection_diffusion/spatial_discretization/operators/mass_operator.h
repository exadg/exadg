#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operation_base.h"

namespace ConvDiff {
    
    
struct MassMatrixOperatorData
{
  MassMatrixOperatorData ()
    :
    dof_index(0),
    quad_index(0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;
};

template <int dim, int fe_degree, typename value_type>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim,fe_degree,value_type> This;

  MassMatrixOperator()
    :
    data(nullptr)
  {}

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  MassMatrixOperatorData const     &mass_matrix_operator_data_in)
  {
    this->data = &mf_data;
    this->mass_matrix_operator_data = mass_matrix_operator_data_in;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src) const
  {
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type> src_dummy(diagonal);

    data->cell_loop(&This::cell_loop_diagonal, this, diagonal, src_dummy);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type> src;

    data->cell_loop(&This::cell_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  MassMatrixOperatorData const & get_operator_data() const
  {
    return mass_matrix_operator_data;
  }


private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (true,false,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value (fe_eval.get_value(q), q);
    }
    fe_eval.integrate (true,false);
  }

  void cell_loop (MatrixFree<dim,value_type> const                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src,
                  std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_diagonal (MatrixFree<dim,value_type> const                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           parallel::distributed::Vector<value_type> const  &,
                           std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (MatrixFree<dim,value_type> const                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  parallel::distributed::Vector<value_type> const  &,
                                                  std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  MassMatrixOperatorData mass_matrix_operator_data;
};
    
}

#endif