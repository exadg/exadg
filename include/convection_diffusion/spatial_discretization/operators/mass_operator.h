#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operator_base.h"

namespace ConvDiff
{
namespace Operators
{
struct MassMatrixKernelData
{
  MassMatrixKernelData() : scaling_factor(1.0)
  {
  }

  double scaling_factor;
};

template<int dim, typename Number>
class MassMatrixKernel
{
public:
  typedef VectorizedArray<Number> scalar;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;

  void
  reinit(MassMatrixKernelData const & data_in) const
  {
    data = data_in;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(true, false, false);
    flags.cell_integrate = CellFlags(true, false, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_JxW_values;

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    return data.scaling_factor * integrator.get_value(q);
  }

private:
  mutable MassMatrixKernelData data;
};

} // namespace Operators


struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::MassMatrixKernelData kernel_data;
};

template<int dim, typename Number>
class MassMatrixOperator : public OperatorBase<dim, Number, MassMatrixOperatorData>
{
private:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData> Base;

  typedef typename Base::IntegratorCell IntegratorCell;

public:
  MassMatrixOperator()
  {
  }

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         MassMatrixOperatorData const &    operator_data) const
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    kernel.reinit(operator_data.kernel_data);

    this->integrator_flags = kernel.get_integrator_flags();
  }

private:
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_value(kernel.get_volume_flux(integrator, q), q);
  }

  Operators::MassMatrixKernel<dim, Number> kernel;
};
} // namespace ConvDiff

#endif
