#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/mass_matrix_kernel.h"
#include "../../../operators/operator_base.h"

#include "../../user_interface/boundary_descriptor.h"

namespace ConvDiff
{
template<int dim>
struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  // required by OperatorBase interface
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData<dim>> Base;

  typedef typename Base::IntegratorCell Integrator;

public:
  MassMatrixOperator();

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         AffineConstraints<double> const &   constraint_matrix,
         MassMatrixOperatorData<dim> const & data);

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
