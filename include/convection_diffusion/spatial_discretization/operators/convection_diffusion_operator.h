#ifndef CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR
#define CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

#include "convection_operator.h"
#include "diffusive_operator.h"
#include "mass_operator.h"

#include "../../../operators/elementwise_operator.h"
#include "../../../solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "../../../solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

namespace ConvDiff
{
template<int dim>
struct ConvectionDiffusionOperatorData : public OperatorBaseData<dim>
{
  ConvectionDiffusionOperatorData()
    : OperatorBaseData<dim>(0, 0),
      unsteady_problem(true),
      convective_problem(true),
      diffusive_problem(true),
      scaling_factor_time_derivative_term(-1.0),
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2)),
      mg_operator_type(MultigridOperatorType::Undefined)
  {
  }

  void
  update_mapping_update_flags()
  {
    if(unsteady_problem)
      this->append_mapping_update_flags(mass_matrix_operator_data);
    if(convective_problem)
      this->append_mapping_update_flags(convective_operator_data);
    if(diffusive_problem)
      this->append_mapping_update_flags(diffusive_operator_data);
  }

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;

  double scaling_factor_time_derivative_term;

  MassMatrixOperatorData<dim> mass_matrix_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
  DiffusiveOperatorData<dim>  diffusive_operator_data;

  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;

  MultigridOperatorType mg_operator_type;

  // TODO: do we really need this here because the convective and diffusive operators already have
  // it
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, int degree, typename Number = double>
class ConvectionDiffusionOperator
  : public OperatorBase<dim, degree, Number, ConvectionDiffusionOperatorData<dim>>
{
public:
  // TODO: Issue#2
  static const int DIM = dim;

  typedef Number value_type;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ConvectionDiffusionOperator<dim, degree, Number> This;

  typedef OperatorBase<dim, degree, value_type, ConvectionDiffusionOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;

  typedef typename Parent::BlockMatrix BlockMatrix;

#ifdef DEAL_II_WITH_TRILINOS
  typedef typename Parent::SparseMatrix SparseMatrix;
#endif

  ConvectionDiffusionOperator();

  void
  initialize(MatrixFree<dim, Number> const &                 mf_data_in,
             ConvectionDiffusionOperatorData<dim> const &    operator_data_in,
             MassMatrixOperator<dim, degree, Number> const & mass_matrix_operator_in,
             ConvectiveOperator<dim, degree, Number> const & convective_operator_in,
             DiffusiveOperator<dim, degree, Number> const &  diffusive_operator_in);


  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. Own operators (mass, convection, diffusion) are
   *  created.
   */
  void
  reinit(DoFHandler<dim> const &   dof_handler,
         Mapping<dim> const &      mapping,
         void *                    operator_data_in,
         MGConstrainedDoFs const & mg_constrained_dofs,
         unsigned int const        level);

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

  // Apply matrix-vector multiplication.
  void
  vmult(VectorType & dst, VectorType const & src) const;

  void
  vmult_add(VectorType & dst, VectorType const & src) const;

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  calculate_system_matrix(SparseMatrix & system_matrix, Number const time) const;

  virtual void
  calculate_system_matrix(SparseMatrix & system_matrix) const;
#endif

  /*
   * This function calculates the diagonal.
   */
  void
  calculate_diagonal(VectorType & diagonal) const;

  /*
   * Block diagonal preconditioner.
   */

  // apply the inverse block diagonal operator (for matrix-based and matrix-free variants)
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;

  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src,
                                       unsigned int const problem_size = 1) const;

private:
  /*
   * This function calculates the block Jacobi matrices and adds the result to matrices. This is
   * done sequentially for the different operators.
   */
  void
  add_block_diagonal_matrices(BlockMatrix & matrices) const;

  void
  add_block_diagonal_matrices(BlockMatrix & matrices, Number const time) const;

  void
  initialize_block_diagonal_preconditioner_matrix_free() const;

  MultigridOperatorBase<dim, Number> *
  get_new(unsigned int deg) const;

  mutable lazy_ptr<MassMatrixOperator<dim, degree, Number>> mass_matrix_operator;
  mutable lazy_ptr<ConvectiveOperator<dim, degree, Number>> convective_operator;
  mutable lazy_ptr<DiffusiveOperator<dim, degree, Number>>  diffusive_operator;

  mutable VectorType temp;
  double             scaling_factor_time_derivative_term;

  // Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
  typedef Elementwise::OperatorBase<dim, Number, This>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PRECONDITIONER_BASE;
  typedef Elementwise::IterativeSolver<dim,
                                       1 /*scalar equation*/,
                                       degree,
                                       Number,
                                       ELEMENTWISE_OPERATOR,
                                       PRECONDITIONER_BASE>
    ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR> elementwise_operator;
  mutable std::shared_ptr<PRECONDITIONER_BASE>  elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>   elementwise_solver;
};

} // namespace ConvDiff

#endif
