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

  // reinit mass, convective and diffusive operators
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
  this->mass_matrix_operator.reset(mass_matrix_operator);
  this->convective_operator.reset(convective_operator);
  this->diffusive_operator.reset(diffusive_operator);

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
  double const & factor) const
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
ConvectionDiffusionOperator<dim, Number>::set_velocity_copy(VectorType const & velocity) const
{
  convective_operator->set_velocity_copy(velocity);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  convective_operator->set_velocity_ptr(velocity);
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
    convective_operator->set_evaluation_time(this->eval_time);
    convective_operator->apply_add(dst, src);
  }
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
    convective_operator->set_evaluation_time(this->eval_time);
    convective_operator->apply_add(dst, src);
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
ConvectionDiffusionOperator<dim, Number>::calculate_system_matrix(
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

    mass_matrix_operator->calculate_system_matrix(system_matrix);
    system_matrix *= scaling_factor_time_derivative_term;
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->calculate_system_matrix(system_matrix);
  }

  if(this->operator_data.convective_problem == true)
  {
    convective_operator->set_evaluation_time(this->eval_time);
    convective_operator->calculate_system_matrix(system_matrix);
  }
}
#endif

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
    convective_operator->set_evaluation_time(this->eval_time);
    convective_operator->add_diagonal(diagonal);
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
    bool update_preconditioner = false;
    elementwise_solver->solve(dst, src, update_preconditioner);
  }
  else // matrix-based
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
      new INVERSE_MASS(this->get_matrix_free(), this->get_dof_index(), this->get_quad_index()));
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

    mass_matrix_operator->apply_add_block_diagonal_elementwise(cell, dst, src, problem_size);

    Elementwise::scale(dst, scaling_factor_time_derivative_term, problem_size);
  }

  if(this->operator_data.diffusive_problem == true)
  {
    diffusive_operator->apply_add_block_diagonal_elementwise(cell, dst, src, problem_size);
  }

  if(this->operator_data.convective_problem == true)
  {
    Number const time = this->get_evaluation_time();

    convective_operator->set_evaluation_time(time);
    convective_operator->apply_add_block_diagonal_elementwise(cell, dst, src, problem_size);
  }
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

    convective_operator->set_evaluation_time(time);
    convective_operator->add_block_diagonal_matrices(matrices);
  }
}

template class ConvectionDiffusionOperator<2, float>;
template class ConvectionDiffusionOperator<2, double>;

template class ConvectionDiffusionOperator<3, float>;
template class ConvectionDiffusionOperator<3, double>;

} // namespace ConvDiff
