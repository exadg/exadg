#include "momentum_operator.h"

#include <deal.II/distributed/tria.h>

#include "functionalities/categorization.h"

#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"

namespace IncNS
{
template<int dim, typename Number>
MomentumOperator<dim, Number>::MomentumOperator()
  : data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(-1.0),
    block_diagonal_preconditioner_is_initialized(false)
{
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_multigrid(MatrixFree<dim, Number> const &   data,
                                                AffineConstraints<double> const & constraint_matrix,
                                                MomentumOperatorData<dim> const & operator_data)
{
  (void)constraint_matrix;

  // setup own mass matrix operator
  own_mass_matrix_operator_storage.initialize(data, operator_data.mass_matrix_operator_data);

  // TODO: refactor viscous operator, s.t. it does not need mapping
  unsigned int const   degree = operator_data.viscous_operator_data.degree;
  MappingQGeneric<dim> mapping(degree);
  own_viscous_operator_storage.initialize(mapping, data, operator_data.viscous_operator_data);
  own_convective_operator_storage.initialize(data, operator_data.convective_operator_data);

  this->reinit(data,
               operator_data,
               own_mass_matrix_operator_storage,
               own_viscous_operator_storage,
               own_convective_operator_storage);

  // initialize temp-vector: this is done in this function because
  // the vector temp is only used in the function vmult_add(), i.e.,
  // when using the multigrid preconditioner
  this->initialize_dof_vector(temp_vector);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &         data,
                                      MomentumOperatorData<dim> const &       operator_data,
                                      MassMatrixOperator<dim, Number> const & mass_matrix_operator,
                                      ViscousOperator<dim, Number> const &    viscous_operator,
                                      ConvectiveOperator<dim, Number> const & convective_operator)
{
  // copy parameters into element variables
  this->data                 = &data;
  this->operator_data        = operator_data;
  this->mass_matrix_operator = &mass_matrix_operator;
  this->viscous_operator     = &viscous_operator;
  this->convective_operator  = &convective_operator;

  // mass matrix term: set scaling factor time derivative term (initialization only, has to be
  // updated in case of time-dependent coefficients)
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_scaling_factor_time_derivative_term(double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, typename Number>
double
MomentumOperator<dim, Number>::get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_solution_linearization(VectorType const & solution_linearization)
{
  if(operator_data.convective_problem == true)
  {
    convective_operator->set_solution_linearization(solution_linearization);
  }
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
MomentumOperator<dim, Number>::get_solution_linearization() const
{
  AssertThrow(operator_data.convective_problem == true,
              ExcMessage(
                "Attempt to access velocity_linearization which has not been initialized."));

  return convective_operator->get_solution_linearization();
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_evaluation_time(double const evaluation_time_in)
{
  evaluation_time = evaluation_time_in;
}

template<int dim, typename Number>
double
MomentumOperator<dim, Number>::get_evaluation_time() const
{
  return evaluation_time;
}

template<int dim, typename Number>
MomentumOperatorData<dim> const &
MomentumOperator<dim, Number>::get_operator_data() const
{
  return this->operator_data;
}

template<int dim, typename Number>
MassMatrixOperatorData const &
MomentumOperator<dim, Number>::get_mass_matrix_operator_data() const
{
  return mass_matrix_operator->get_operator_data();
}

template<int dim, typename Number>
ConvectiveOperatorData<dim> const &
MomentumOperator<dim, Number>::get_convective_operator_data() const
{
  return convective_operator->get_operator_data();
}

template<int dim, typename Number>
ViscousOperatorData<dim> const &
MomentumOperator<dim, Number>::get_viscous_operator_data() const
{
  return viscous_operator->get_operator_data();
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
MomentumOperator<dim, Number>::get_data() const
{
  return *data;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(
      this->get_scaling_factor_time_derivative_term() > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, evaluation_time);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::vmult_add(VectorType & dst, VectorType const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply(temp_vector, src);
    temp_vector *= this->get_scaling_factor_time_derivative_term();
    dst += temp_vector;
  }

  viscous_operator->apply_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, evaluation_time);
  }
}


template<int dim, typename Number>
void
MomentumOperator<dim, Number>::vmult_block_jacobi(VectorType & dst, VectorType const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_block_diagonal_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_block_diagonal_add(dst, src, evaluation_time);
  }
}

template<int dim, typename Number>
unsigned int
MomentumOperator<dim, Number>::get_dof_index() const
{
  return operator_data.dof_index;
}

template<int dim, typename Number>
unsigned int
MomentumOperator<dim, Number>::get_quad_index() const
{
  return operator_data.quad_index_std;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  calculate_diagonal(diagonal);

  // verify_calculation_of_diagonal(*this,diagonal);

  invert_diagonal(diagonal);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::calculate_diagonal(VectorType & diagonal) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    diagonal = 0.0;
  }

  viscous_operator->add_diagonal(diagonal);

  if(operator_data.convective_problem == true)
  {
    convective_operator->add_diagonal(diagonal, evaluation_time);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::initialize_block_diagonal_preconditioner_matrix_free() const
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
    typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, Number> INVERSE_MASS;

    elementwise_preconditioner.reset(
      new INVERSE_MASS(this->get_data(), this->get_dof_index(), this->get_quad_index()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  if(operator_data.convective_problem)
    iterative_solver_data.solver_type = Elementwise::SolverType::GMRES;
  else
    iterative_solver_data.solver_type = Elementwise::SolverType::CG;

  iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

  elementwise_solver.reset(new ELEMENTWISE_SOLVER(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
    iterative_solver_data));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::update_block_diagonal_preconditioner() const
{
  // initialization

  if(!block_diagonal_preconditioner_is_initialized)
  {
    if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      initialize_block_diagonal_preconditioner_matrix_free();
    }
    else // matrix-based variant
    {
      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = data->get_shape_info().dofs_per_component_on_cell * dim;

      matrices.resize(data->n_macro_cells() * VectorizedArray<value_type>::n_array_elements,
                      LAPACKFullMatrix<value_type>(dofs_per_cell, dofs_per_cell));
    }

    block_diagonal_preconditioner_is_initialized = true;
  }

  // update

  // For the matrix-free variant there is nothing to do.
  // For the matrix-based variant we have to recompute the block matrices.
  if(!operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // clear matrices
    initialize_block_jacobi_matrices_with_zero(matrices);

    // compute block matrices and add
    this->add_block_diagonal_matrices(matrices);

    // check_block_jacobi_matrices();

    calculate_lu_factorization_block_jacobi(matrices);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::apply_inverse_block_diagonal(VectorType &       dst,
                                                            VectorType const & src) const
{
  // matrix-free
  if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // Solve block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    bool variable_not_needed = false;
    elementwise_solver->solve(dst, src, variable_not_needed);
  }
  else // matrix based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    data->cell_loop(&This::cell_loop_apply_inverse_block_diagonal, this, dst, src);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::add_block_diagonal_matrices(
  std::vector<LAPACKFullMatrix<value_type>> & matrices) const
{
  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->add_block_diagonal_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin();
        it != matrices.end();
        ++it)
    {
      (*it) *= this->get_scaling_factor_time_derivative_term();
    }
  }

  viscous_operator->add_block_diagonal_matrices(matrices);

  if(operator_data.convective_problem == true)
  {
    convective_operator->add_block_diagonal_matrices(matrices, evaluation_time);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src,
  unsigned int const                    problem_size) const
{
  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply_add_block_diagonal_elementwise(cell, dst, src);

    Elementwise::scale(dst, this->get_scaling_factor_time_derivative_term(), problem_size);
  }

  viscous_operator->apply_add_block_diagonal_elementwise(cell, dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_add_block_diagonal_elementwise(cell, dst, src);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::cell_loop_apply_inverse_block_diagonal(
  MatrixFree<dim, Number> const &               data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  Integrator integrator(data, operator_data.dof_index, operator_data.quad_index_std);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    unsigned int dofs_per_cell = integrator.dofs_per_cell;

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = integrator.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * VectorizedArray<Number>::n_array_elements + v].solve(src_vector, false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j][v] = src_vector(j);
    }

    integrator.set_dof_values(dst);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::check_block_jacobi_matrices() const
{
  VectorType src;
  this->initialize_dof_vector(src);

  // fill vector with values unequal zero
  for(unsigned int i = 0; i < src.local_size(); ++i)
    src.local_element(i) = i % 11;

  // test matrix-vector product for block Jacobi problem by comparing
  // matrix-free matrix-vector product and matrix-based matrix-vector product
  // (where the matrices are generated using the matrix-free implementation)
  VectorType tmp1(src), tmp2(src), diff(src);
  tmp1 = 0.0;
  tmp2 = 0.0;

  // variant 1 (matrix-free)
  vmult_block_jacobi(tmp1, src);

  // variant 2 (matrix-based)
  data->cell_loop(&This::cell_loop_apply_block_diagonal, this, tmp2, src);

  diff = tmp2;
  diff.add(-1.0, tmp1);

  std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
            << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
            << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl
            << std::endl;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::cell_loop_apply_block_diagonal(
  MatrixFree<dim, Number> const &               data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  Integrator integrator(data, operator_data.dof_index, operator_data.quad_index_std);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    unsigned int dofs_per_cell = integrator.dofs_per_cell;

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(dofs_per_cell);
      Vector<Number> dst_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = integrator.begin_dof_values()[j][v];

      // apply matrix-vector product
      matrices[cell * VectorizedArray<Number>::n_array_elements + v].vmult(dst_vector,
                                                                           src_vector,
                                                                           false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j][v] = dst_vector(j);
    }

    integrator.set_dof_values(dst);
  }
}

template class MomentumOperator<2, float>;
template class MomentumOperator<2, double>;

template class MomentumOperator<3, float>;
template class MomentumOperator<3, double>;

} // namespace IncNS
