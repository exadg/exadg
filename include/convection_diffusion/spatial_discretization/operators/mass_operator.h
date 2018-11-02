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
public:
  typedef MassMatrixOperator<dim, degree, Number> This;

  typedef OperatorBase<dim, degree, Number, MassMatrixOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;

  MassMatrixOperator()
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             MassMatrixOperatorData<dim> const & mass_matrix_operator_data,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             ConstraintMatrix const &            constraint_matrix,
             MassMatrixOperatorData<dim> const & mass_matrix_operator_data,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

private:
  void
  do_cell_integral(FEEvalCell & fe_eval) const;
};
} // namespace ConvDiff

#endif
