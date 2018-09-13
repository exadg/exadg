#ifndef INCLUDE_CONVECTION_DIFFUSION_RHS
#define INCLUDE_CONVECTION_DIFFUSION_RHS


#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../../include/functionalities/evaluate_functions.h"

namespace ConvDiff
{

template<int dim>
struct RHSOperatorData
{
  RHSOperatorData ()
    :
    dof_index(0),
    quad_index(0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;
  std::shared_ptr<Function<dim> > rhs;
};

template <int dim, int fe_degree, typename value_type>
class RHSOperator
{
public:
  typedef RHSOperator<dim,fe_degree, value_type> This;

  RHSOperator()
    :
    data(nullptr),
    eval_time(0.0)
  {}

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  RHSOperatorData<dim> const       &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  // apply matrix vector multiplication
  void evaluate (parallel::distributed::Vector<value_type> &dst,
                 value_type const                          evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type> &dst,
                     value_type const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
      VectorizedArray<value_type> rhs = make_vectorized_array<value_type>(0.0);
      evaluate_scalar_function(rhs,operator_data.rhs,q_points,eval_time);

      fe_eval.submit_value (rhs, q);
    }
    fe_eval.integrate (true,false);
  }

  void cell_loop (MatrixFree<dim,value_type> const                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &,
                  std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  RHSOperatorData<dim> operator_data;
  value_type mutable eval_time;
};

}

#endif