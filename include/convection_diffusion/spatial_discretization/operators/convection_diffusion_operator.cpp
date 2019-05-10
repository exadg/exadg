#include "convection_diffusion_operator.h"

#include "../../../functionalities/constraints.h"
#include "solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

namespace ConvDiff
{
template<int dim, typename Number>
ConvectionDiffusionOperator<dim, Number>::ConvectionDiffusionOperator()
  : scaling_factor_time_derivative_term(-1.0)
{
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &              mf_data,
  AffineConstraints<double> const &            constraint_matrix,
  ConvectionDiffusionOperatorData<dim> const & operator_data) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);

  // mass matrix term: set scaling factor time derivative term
  this->scaling_factor_time_derivative_term =
    this->operator_data.scaling_factor_time_derivative_term;

  // use own data structures (TODO: no switch necessary)
  this->mass_matrix_operator.reset();
  this->convective_operator.reset();
  this->diffusive_operator.reset();

  // reinit mass-, convection- and diffusive-opertor
  this->mass_matrix_operator->reinit(mf_data,
                                     constraint_matrix,
                                     operator_data.mass_matrix_operator_data);
  this->convective_operator->reinit(mf_data,
                                    constraint_matrix,
                                    operator_data.convective_operator_data);
  this->diffusive_operator->reinit(mf_data,
                                   constraint_matrix,
                                   operator_data.diffusive_operator_data);

  // initialize temp-vector: this is done in this function because
  // the vector temp is only used in the function vmult_add(), i.e.,
  // when using the multigrid preconditioner
  this->initialize_dof_vector(temp);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &              mf_data,
  AffineConstraints<double> const &            constraint_matrix,
  ConvectionDiffusionOperatorData<dim> const & operator_data,
  MassMatrixOperator<dim, Number> const &      mass_matrix_operator,
  ConvectiveOperator<dim, Number> const &      convective_operator,
  DiffusiveOperator<dim, Number> const &       diffusive_operator) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);
  this->mass_matrix_operator.reinit(mass_matrix_operator);
  this->convective_operator.reinit(convective_operator);
  this->diffusive_operator.reinit(diffusive_operator);

  // mass matrix term: set scaling factor time derivative term
  this->scaling_factor_time_derivative_term =
    this->operator_data.scaling_factor_time_derivative_term;

  // initialize temp-vector: this is done in this function because
  // the vector temp is only used in the function vmult_add(), i.e.,
  // when using the multigrid preconditioner
  this->initialize_dof_vector(temp);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::set_scaling_factor_time_derivative_term(
  double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, typename Number>
double
ConvectionDiffusionOperator<dim, Number>::get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, typename Number>
AffineConstraints<double> const &
ConvectionDiffusionOperator<dim, Number>::get_constraint_matrix() const
{
  return this->do_get_constraint_matrix();
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
ConvectionDiffusionOperator<dim, Number>::get_data() const
{
  return *this->data;
}

template<int dim, typename Number>
unsigned int
ConvectionDiffusionOperator<dim, Number>::get_dof_index() const
{
  return this->operator_data.dof_index;
}

template<int dim, typename Number>
unsigned int
ConvectionDiffusionOperator<dim, Number>::get_quad_index() const
{
  return this->operator_data.quad_index;
}

template<int dim, typename Number>
std::shared_ptr<BoundaryDescriptor<dim>>
ConvectionDiffusionOperator<dim, Number>::get_boundary_descriptor() const
{
  return diffusive_operator->get_operator_data().bc;
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
ConvectionDiffusionOperator<dim, Number>::get_velocity() const
{
  return convective_operator->get_velocity();
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::set_velocity(VectorType const & velocity) const
{
  convective_operator->set_velocity(velocity);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  this->apply(dst, src);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::vmult_add(VectorType & dst, VectorType const & src) const
{
  this->apply_add(dst, src);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been initialized!"));

    mass_matrix_operator->apply(dst, src);
    dst *= scaling_factor_time_derivative_term;
  }
  else
  {
    dst = 0.0;
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->apply_add(dst, src);
  }

  if(this->operator_data.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, this->eval_time);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::apply_add(VectorType &       dst,
                                                    VectorType const & src,
                                                    Number const       time) const
{
  Base::apply_add(dst, src, time);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(
      scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->apply(temp, src);
    temp *= scaling_factor_time_derivative_term;
    dst += temp;
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->apply_add(dst, src);
  }

  if(this->operator_data.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, this->eval_time);
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::init_system_matrix(SparseMatrix & system_matrix) const
{
  this->do_init_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::calculate_system_matrix(SparseMatrix & system_matrix,
                                                                  Number const   time) const
{
  this->eval_time = time;
  calculate_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::calculate_system_matrix(
  SparseMatrix & system_matrix) const
{
  this->do_calculate_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::do_calculate_system_matrix(SparseMatrix & system_matrix,
                                                                     Number const   time) const
{
  this->eval_time = time;
  do_calculate_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::do_calculate_system_matrix(
  SparseMatrix & system_matrix) const
{
  // clear content of matrix since the next calculate_system_matrix-commands add their result
  system_matrix *= 0.0;

  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(
      scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->do_calculate_system_matrix(system_matrix);
    system_matrix *= scaling_factor_time_derivative_term;
  }
  else
  {
    // nothing to do since matrix is already explicitly set to zero
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->do_calculate_system_matrix(system_matrix);
  }

  if(this->operator_data.convective_problem == true)
  {
    convective_operator->do_calculate_system_matrix(system_matrix, this->eval_time);
  }
}
#endif

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::calculate_inverse_diagonal(
  VectorType & inverse_diagonal) const
{
  calculate_diagonal(inverse_diagonal);

  //   verify_calculation_of_diagonal(*this,inverse_diagonal);

  invert_diagonal(inverse_diagonal);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::calculate_diagonal(VectorType & diagonal) const
{
  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(
      scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= scaling_factor_time_derivative_term;
  }
  else
  {
    diagonal = 0.0;
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->add_diagonal(diagonal);
  }

  if(this->operator_data.convective_problem == true)
  {
    convective_operator->add_diagonal(diagonal, this->eval_time);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::apply_inverse_block_diagonal(VectorType &       dst,
                                                                       VectorType const & src) const
{
  // matrix-free
  if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // Solve elementwise block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    bool variable_not_needed = false;
    elementwise_solver->solve(dst, src, variable_not_needed);
  }
  else // matrix based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    Base::apply_inverse_block_diagonal_matrix_based(dst, src);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::initialize_block_diagonal_preconditioner_matrix_free()
  const
{
  elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

  if(this->operator_data.preconditioner_block_jacobi == PreconditionerBlockDiagonal::None)
  {
    typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
    elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
  }
  else if(this->operator_data.preconditioner_block_jacobi ==
          PreconditionerBlockDiagonal::InverseMassMatrix)
  {
    typedef Elementwise::InverseMassMatrixPreconditioner<dim, 1 /*scalar equation*/, Number>
      INVERSE_MASS;

    elementwise_preconditioner.reset(
      new INVERSE_MASS(this->get_data(), this->get_dof_index(), this->get_quad_index()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  iterative_solver_data.solver_type = Elementwise::SolverType::GMRES;
  iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

  elementwise_solver.reset(new ELEMENTWISE_SOLVER(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
    iterative_solver_data));
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::update_block_diagonal_preconditioner() const
{
  this->do_update_block_diagonal_preconditioner();
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src,
  unsigned int const                    problem_size) const
{
  // calculate block Jacobi matrices
  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been initialized!"));

    mass_matrix_operator->apply_add_block_diagonal_elementwise(cell, dst, src);

    Elementwise::scale(dst, scaling_factor_time_derivative_term, problem_size);
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->apply_add_block_diagonal_elementwise(cell, dst, src);
  }

  if(this->operator_data.convective_problem == true)
  {
    Number const time = this->get_evaluation_time();

    convective_operator->apply_add_block_diagonal_elementwise(cell, dst, src, time);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::add_block_diagonal_matrices(BlockMatrix & /* matrices */,
                                                                      Number const /* time */) const
{
  // We have to override this function but do not need it for this operator.
  AssertThrow(false, ExcMessage("Should not arrive here."));
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::add_block_diagonal_matrices(BlockMatrix & matrices) const
{
  // calculate block Jacobi matrices
  if(this->operator_data.unsteady_problem == true)
  {
    AssertThrow(scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been initialized!"));

    mass_matrix_operator->add_block_diagonal_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin();
        it != matrices.end();
        ++it)
    {
      (*it) *= scaling_factor_time_derivative_term;
    }
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->add_block_diagonal_matrices(matrices);
  }

  if(this->operator_data.convective_problem == true)
  {
    Number const time = this->get_evaluation_time();

    convective_operator->add_block_diagonal_matrices(matrices, time);
  }
}

// TODO
template<int dim, typename Number>
PreconditionableOperator<dim, Number> *
ConvectionDiffusionOperator<dim, Number>::get_new(unsigned int /*deg*/) const
{
  return new ConvectionDiffusionOperator<dim, Number>();

  //  switch(deg)
  //  {
  //#if DEGREE_1
  //    case 1:
  //      return new ConvectionDiffusionOperator<dim, 1, Number>();
  //#endif
  //#if DEGREE_2
  //    case 2:
  //      return new ConvectionDiffusionOperator<dim, 2, Number>();
  //#endif
  //#if DEGREE_3
  //    case 3:
  //      return new ConvectionDiffusionOperator<dim, 3, Number>();
  //#endif
  //#if DEGREE_4
  //    case 4:
  //      return new ConvectionDiffusionOperator<dim, 4, Number>();
  //#endif
  //#if DEGREE_5
  //    case 5:
  //      return new ConvectionDiffusionOperator<dim, 5, Number>();
  //#endif
  //#if DEGREE_6
  //    case 6:
  //      return new ConvectionDiffusionOperator<dim, 6, Number>();
  //#endif
  //#if DEGREE_7
  //    case 7:
  //      return new ConvectionDiffusionOperator<dim, 7, Number>();
  //#endif
  //#if DEGREE_8
  //    case 8:
  //      return new ConvectionDiffusionOperator<dim, 8, Number>();
  //#endif
  //#if DEGREE_9
  //    case 9:
  //      return new ConvectionDiffusionOperator<dim, 9, Number>();
  //#endif
  //#if DEGREE_10
  //    case 10:
  //      return new ConvectionDiffusionOperator<dim, 10, Number>();
  //#endif
  //#if DEGREE_11
  //    case 11:
  //      return new ConvectionDiffusionOperator<dim, 11, Number>();
  //#endif
  //#if DEGREE_12
  //    case 12:
  //      return new ConvectionDiffusionOperator<dim, 12, Number>();
  //#endif
  //#if DEGREE_13
  //    case 13:
  //      return new ConvectionDiffusionOperator<dim, 13, Number>();
  //#endif
  //#if DEGREE_14
  //    case 14:
  //      return new ConvectionDiffusionOperator<dim, 14, Number>();
  //#endif
  //#if DEGREE_15
  //    case 15:
  //      return new ConvectionDiffusionOperator<dim, 15, Number>();
  //#endif
  //    default:
  //      AssertThrow(false,
  //                  ExcMessage("ConvectionDiffusionOperator not implemented for this degree!"));
  //      return nullptr;
  //  }
}

template class ConvectionDiffusionOperator<2, float>;
template class ConvectionDiffusionOperator<2, double>;

template class ConvectionDiffusionOperator<3, float>;
template class ConvectionDiffusionOperator<3, double>;

} // namespace ConvDiff
