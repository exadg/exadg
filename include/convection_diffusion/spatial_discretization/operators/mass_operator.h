#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"

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

template<int dim, int degree, typename Number>
class MassMatrixOperator : public OperatorBase<dim, degree, Number, MassMatrixOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, degree, Number, MassMatrixOperatorData<dim>> Base;

  typedef typename Base::FEEvalCell FEEvalCell;

public:
  static const int                  DIM = dim;
  typedef typename Base::VectorType VectorType;

private:
  void
  do_cell_integral(FEEvalCell & fe_eval, unsigned int const /*cell*/) const;
};
} // namespace ConvDiff

#endif
