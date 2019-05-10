#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operator_base.h"

namespace ConvDiff
{
template<int dim>
struct MassMatrixOperatorData : public OperatorBaseData<dim>
{
  MassMatrixOperatorData()
    // clang-format off
    : OperatorBaseData<dim>(0, 0,
          true, false, false, true, false, false) // cell
  // clang-format on
  {
    this->mapping_update_flags = update_values | update_quadrature_points;
  }
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData<dim>>
{
public:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData<dim>> Base;

  typedef typename Base::VectorType VectorType;

private:
  typedef typename Base::FEEvalCell FEEvalCell;

  void
  do_cell_integral(FEEvalCell & integrator, unsigned int const /*cell*/) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_value(integrator.get_value(q), q);
  }
};
} // namespace ConvDiff

#endif
