#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/mass_matrix_kernel.h"
#include "../../../operators/operator_base.h"

namespace ConvDiff
{
struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData>
{
private:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData> Base;

  typedef typename Base::IntegratorCell Integrator;

public:
  MassMatrixOperator();

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         MassMatrixOperatorData const &    data);

  void
  set_scaling_factor(Number const & number);

private:
  void
  do_cell_integral(Integrator & integrator) const;

  MassMatrixKernel<dim, Number> kernel;

  double scaling_factor;
};
} // namespace ConvDiff

#endif
