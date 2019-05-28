#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operator_base.h"

namespace ConvDiff
{
struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData()
    // clang-format off
    : OperatorBaseData(
          0, // dof_index
          0, // quad_index
          true, false, false, // cell evaluate
          true, false, false) // cell integrate
  // clang-format on
  {
    this->mapping_update_flags = update_values | update_quadrature_points;
  }
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData>
{
public:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData> Base;

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
