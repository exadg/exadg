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

  MassMatrixKernel() : scaling_factor(1.0)
  {
  }

  void
  reinit(double const & factor) const
  {
    set_scaling_factor(factor);
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

  Number
  get_scaling_factor() const
  {
    return scaling_factor;
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

  typedef typename Base::IntegratorCell Integrator;

public:
  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         MassMatrixOperatorData const &    data) const;

  void
  set_scaling_factor(Number const & number);

private:
  void
  do_cell_integral(Integrator & integrator) const;

  Operators::MassMatrixKernel<dim, Number> kernel;
};
} // namespace ConvDiff

#endif
