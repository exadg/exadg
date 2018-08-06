#ifndef CONV_DIFF_MASS_OPERATOR
#define CONV_DIFF_MASS_OPERATOR

#include "../../../operators/operation_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../types.h"


namespace ConvDiff
{
template<int dim>
struct MassMatrixOperatorData
  : public OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>>
{
  MassMatrixOperatorData()
    : OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>>(0,
                                                                                           0,
                                                                                           true,
                                                                                           false,
                                                                                           false,
                                                                                           true,
                                                                                           false,
                                                                                           false)
  {
  }
};

template<int dim, int fe_degree, typename value_type>
class MassMatrixOperator : public OperatorBase<dim, fe_degree, value_type, MassMatrixOperatorData<dim>>
{
public:
  typedef MassMatrixOperator<dim, fe_degree, value_type>                        This;
  typedef OperatorBase<dim, fe_degree, value_type, MassMatrixOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell                                           FEEvalCell;
  typedef typename Parent::FEEvalFace                                           FEEvalFace;

  MassMatrixOperator()
  {
  }

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             MassMatrixOperatorData<dim> const & mass_matrix_operator_data_in);

private:
  void
  do_cell_integral(FEEvalCell & fe_eval) const;

  void
  do_face_integral(FEEvalFace &, FEEvalFace &) const
  {
  }

  void
  do_face_int_integral(FEEvalFace &, FEEvalFace &) const
  {
  }

  void
  do_face_ext_integral(FEEvalFace &, FEEvalFace &) const
  {
  }

  void
  do_boundary_integral(FEEvalFace &, OperatorType const &, types::boundary_id const &) const
  {
  }
};
} // namespace ConvDiff

#endif