#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operator_base.h"

namespace ConvDiff
{
namespace Operators
{
template<int dim, typename Number>
class MassMatrixKernel
{
public:
  typedef VectorizedArray<Number> scalar;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;

  MassMatrixKernel() : scaling_factor(1.0)
  {
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

  void
  set_scaling_factor(Number const & number) const
  {
    scaling_factor = number;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(scalar const & value) const
  {
    return scaling_factor * value;
  }

private:
  mutable Number scaling_factor;
};

} // namespace Operators


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

    this->integrator_flags = kernel.get_integrator_flags();
  }

  void
  set_scaling_factor(Number const & number)
  {
    kernel.set_scaling_factor(number);
  }

private:
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(kernel.get_volume_flux(integrator.get_value(q)), q);
    }
  }

  Operators::MassMatrixKernel<dim, Number> kernel;
};
} // namespace ConvDiff

#endif
