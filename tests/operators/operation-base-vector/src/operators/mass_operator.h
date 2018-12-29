#ifndef TEST_VECTOR_MASS_OPERATOR
#define TEST_VECTOR_MASS_OPERATOR

#include "../../../../../include/operators/operator_base.h"

namespace IncNS
{
template<int dim>
struct MassMatrixOperatorDataNew : public OperatorBaseData<dim>
{
  MassMatrixOperatorDataNew()
    // clang-format off
    : OperatorBaseData<dim>(0, 0,
          true, false, false, true, false, false) // cell
  // clang-format on
  {
    this->mapping_update_flags = update_values | update_quadrature_points;
  }
};

template<int dim, int degree, typename Number>
class MassMatrixOperatorNew : public OperatorBase<dim, degree, Number, MassMatrixOperatorDataNew<dim>, dim>
{
private:
  typedef OperatorBase<dim, degree, Number, MassMatrixOperatorDataNew<dim>,dim> Base;

  typedef typename Base::FEEvalCell FEEvalCell;

public:
  static const int                  DIM = dim;
  typedef typename Base::VectorType VectorType;
  
  MassMatrixOperatorNew() : scaling_factor(1.0){}

private:
  void
  do_cell_integral(FEEvalCell & fe_eval, unsigned int const /*cell*/) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_value(scaling_factor * fe_eval.get_value(q), q);
  }

  public:
    mutable Number scaling_factor;
};
}

#endif
