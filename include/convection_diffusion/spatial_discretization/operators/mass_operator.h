#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operation_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../types.h"


namespace ConvDiff
{
template<int dim>
struct MassMatrixOperatorData : public OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>
{
  MassMatrixOperatorData()
    // clang-format off
    : OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>(0, 0,
          true, false, false, true, false, false)
  // clang-format on
  {
    this->mapping_update_flags = update_values | update_quadrature_points;
  }
};

template<int dim, int fe_degree, typename value_type>
class MassMatrixOperator
  : public OperatorBase<dim, fe_degree, value_type, MassMatrixOperatorData<dim>>
{
public:
  typedef MassMatrixOperator<dim, fe_degree, value_type> This;

  typedef OperatorBase<dim, fe_degree, value_type, MassMatrixOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;

  MassMatrixOperator()
  {
  }

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             ConstraintMatrix const &            constraint_matrix,
             MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

private:
  void
  do_cell_integral(FEEvalCell & fe_eval) const;
};
} // namespace ConvDiff

#endif
