#ifndef CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR
#define CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR

#include "../../../operators/operation_base.h"

#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"
#include "../types.h"
#include "convection_operator.h"
#include "diffusive_operator.h"
#include "mass_operator.h"


namespace ConvDiff
{
template<int dim>
struct ConvectionDiffusionOperatorData
  : public OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>>
{
  ConvectionDiffusionOperatorData()
    : OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>>(0, 0),
      unsteady_problem(true),
      convective_problem(true),
      diffusive_problem(true),
      mg_operator_type(MultigridOperatorType::Undefined),
      scaling_factor_time_derivative_term(-1.0)
  {
  }

  bool                  unsteady_problem;
  bool                  convective_problem;
  bool                  diffusive_problem;
  MultigridOperatorType mg_operator_type;

  double scaling_factor_time_derivative_term;

  MassMatrixOperatorData<dim> mass_matrix_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
  DiffusiveOperatorData<dim>  diffusive_operator_data;
};

template<int dim, int fe_degree, typename Number = double>
class ConvectionDiffusionOperator
  : public OperatorBase<dim, fe_degree, Number, ConvectionDiffusionOperatorData<dim>>
{
public:
  // TODO: Issue#2
  typedef Number                                                                         value_type;
  typedef ConvectionDiffusionOperator<dim, fe_degree, Number>                            This;
  static const int                                                                       DIM = dim;
  typedef OperatorBase<dim, fe_degree, value_type, ConvectionDiffusionOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell                                                    FEEvalCell;
  typedef typename Parent::FEEvalFace                                                    FEEvalFace;
  typedef typename Parent::BlockMatrix                                                       BlockMatrix;

  typedef MassMatrixOperator<dim, fe_degree, Number> MassMatrixOp;
  typedef ConvectiveOperator<dim, fe_degree, Number> ConvectiveOp;
  typedef DiffusiveOperator<dim, fe_degree, Number>  DiffusiveOp;

  ConvectionDiffusionOperator();

  void
  initialize(MatrixFree<dim, Number> const &                    mf_data_in,
             ConvectionDiffusionOperatorData<dim> const &       operator_data_in,
             MassMatrixOperator<dim, fe_degree, Number> const & mass_matrix_operator_in,
             ConvectiveOperator<dim, fe_degree, Number> const & convective_operator_in,
             DiffusiveOperator<dim, fe_degree, Number> const &  diffusive_operator_in);


  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. Own operators (mass, convection, diffusion) are
   *  created.
   */
  void
  reinit(const DoFHandler<dim> &   dof_handler,
         const Mapping<dim> &      mapping,
         void *                    operator_data_in,
         const MGConstrainedDoFs & mg_constrained_dofs,
         const unsigned int        level);

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void
  set_scaling_factor_time_derivative_term(double const & factor);

  double
  get_scaling_factor_time_derivative_term() const;

  /*
   *  Operator data of basic operators: mass matrix, convective operator, diffusive operator
   */
  MassMatrixOperatorData<dim> const &
  get_mass_matrix_operator_data() const;

  ConvectiveOperatorData<dim> const &
  get_convective_operator_data() const;

  DiffusiveOperatorData<dim> const &
  get_diffusive_operator_data() const;

  /*
   *  This function does nothing in case of the ConvectionDiffusionOperator.
   *  It is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void
  set_zero_mean_value(parallel::distributed::Vector<Number> & /*vec*/) const;

  // Apply matrix-vector multiplication.
  void
  vmult(parallel::distributed::Vector<Number> & dst, parallel::distributed::Vector<Number> const & src) const;

  void
  vmult_add(parallel::distributed::Vector<Number> &       dst,
            parallel::distributed::Vector<Number> const & src) const;

private:
  /*
   *  This function calculates the diagonal of the scalar reaction-convection-diffusion operator.
   */
  void
  calculate_diagonal(parallel::distributed::Vector<Number> & diagonal) const;

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void
  add_block_jacobi_matrices(BlockMatrix & matrices) const;
  void
  add_block_jacobi_matrices(BlockMatrix & matrices, Number const time) const;

  MultigridOperatorBase<dim, Number> *
  get_new(unsigned int deg) const;

private:
  mutable lazy_ptr<MassMatrixOp> mass_matrix_operator;
  mutable lazy_ptr<ConvectiveOp> convective_operator;
  mutable lazy_ptr<DiffusiveOp>  diffusive_operator;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;
};

} // namespace ConvDiff

#endif