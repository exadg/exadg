#ifndef INCLUDE_CONVECTION_DIFFUSION_RHS
#define INCLUDE_CONVECTION_DIFFUSION_RHS


#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../../include/functionalities/evaluate_functions.h"

namespace ConvDiff
{
template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<Function<dim>> rhs;
};

template<int dim, int degree, typename Number>
class RHSOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef RHSOperator<dim, degree, Number> This;

  RHSOperator() : data(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const & mf_data, RHSOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  // apply matrix vector multiplication
  void
  evaluate(VectorType & dst, double const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, evaluation_time);
  }

  void
  evaluate_add(VectorType & dst, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  template<typename FEEvaluation>
  inline void
  do_cell_integral(FEEvaluation & fe_eval) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar rhs = make_vectorized_array<Number>(0.0);
      evaluate_scalar_function(rhs, operator_data.rhs, q_points, eval_time);

      fe_eval.submit_value(rhs, q);
    }
    fe_eval.integrate(true, false);
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const & /*src*/,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, 1, Number> fe_eval(data,
                                                             operator_data.dof_index,
                                                             operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  RHSOperatorData<dim> operator_data;

  double mutable eval_time;
};

} // namespace ConvDiff

#endif
