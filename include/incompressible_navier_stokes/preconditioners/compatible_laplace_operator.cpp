#include "compatible_laplace_operator.h"

namespace IncNS
{
template<int dim, typename Number>
CompatibleLaplaceOperator<dim, Number>::CompatibleLaplaceOperator()
  : data(nullptr),
    gradient_operator(nullptr),
    divergence_operator(nullptr),
    inv_mass_matrix_operator(nullptr)
{
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::reinit_multigrid(
  MatrixFree<dim, Number> const &            data,
  AffineConstraints<double> const &          constraint_matrix,
  CompatibleLaplaceOperatorData<dim> const & operator_data)
{
  (void)constraint_matrix;

  // setup own gradient operator
  GradientOperatorData<dim> gradient_operator_data = operator_data.gradient_operator_data;
  own_gradient_operator_storage.initialize(data, gradient_operator_data);

  // setup own divergence operator
  DivergenceOperatorData<dim> divergence_operator_data = operator_data.divergence_operator_data;
  own_divergence_operator_storage.initialize(data, divergence_operator_data);

  // setup own inverse mass matrix operator
  // NOTE: use quad_index = 0 since own_matrix_free_storage contains only one quadrature formula
  // (i.e. on would use quad_index = 0 also if quad_index_velocity would be 1 !)
  unsigned int quad_index = 0;
  own_inv_mass_matrix_operator_storage.initialize(data,
                                                  operator_data.degree_u,
                                                  operator_data.dof_index_velocity,
                                                  quad_index);

  // setup compatible Laplace operator
  initialize(data,
             operator_data,
             own_gradient_operator_storage,
             own_divergence_operator_storage,
             own_inv_mass_matrix_operator_storage);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::initialize(
  MatrixFree<dim, Number> const &                     mf_data_in,
  CompatibleLaplaceOperatorData<dim> const &          operator_data_in,
  GradientOperator<dim, Number> const &               gradient_operator_in,
  DivergenceOperator<dim, Number> const &             divergence_operator_in,
  InverseMassMatrixOperator<dim, dim, Number> const & inv_mass_matrix_operator_in)
{
  // copy parameters into element variables
  this->data                     = &mf_data_in;
  this->operator_data            = operator_data_in;
  this->gradient_operator        = &gradient_operator_in;
  this->divergence_operator      = &divergence_operator_in;
  this->inv_mass_matrix_operator = &inv_mass_matrix_operator_in;

  // initialize tmp vector
  initialize_dof_vector_velocity(tmp);
}

template<int dim, typename Number>
bool
CompatibleLaplaceOperator<dim, Number>::is_singular() const
{
  return this->operator_data.operator_is_singular;
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::Tvmult(VectorType & dst, VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::Tvmult_add(VectorType & dst, VectorType const & src) const
{
  vmult_add(dst, src);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::vmult_add(VectorType & dst, VectorType const & src) const
{
  // compatible Laplace operator = B * M^{-1} * B^{T} = (-div) * M^{-1} * grad
  gradient_operator->apply(tmp, src);
  inv_mass_matrix_operator->apply(tmp, tmp);
  // NEGATIVE divergence operator
  tmp *= -1.0;
  divergence_operator->apply_add(dst, tmp);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::vmult_interface_down(VectorType &       dst,
                                                             VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::vmult_add_interface_up(VectorType &       dst,
                                                               VectorType const & src) const
{
  vmult_add(dst, src);
}

template<int dim, typename Number>
types::global_dof_index
CompatibleLaplaceOperator<dim, Number>::m() const
{
  return data->get_vector_partitioner(operator_data.dof_index_pressure)->size();
}

template<int dim, typename Number>
types::global_dof_index
CompatibleLaplaceOperator<dim, Number>::n() const
{
  return data->get_vector_partitioner(operator_data.dof_index_pressure)->size();
}

template<int dim, typename Number>
Number
CompatibleLaplaceOperator<dim, Number>::el(const unsigned int, const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
CompatibleLaplaceOperator<dim, Number>::get_data() const
{
  return *data;
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::calculate_diagonal(VectorType & diagonal) const
{
  std::cout << "Calculation of diagonal for compatible Laplace operator ..." << std::flush;

  // naive implementation of calculation of diagonal (TODO)
  diagonal = 0.0;
  VectorType src(diagonal), dst(diagonal);
  for(unsigned int i = 0; i < diagonal.local_size(); ++i)
  {
    src.local_element(i) = 1.0;
    vmult(dst, src);
    diagonal.local_element(i) = dst.local_element(i);
    src.local_element(i)      = 0.0;
  }

  std::cout << " done." << std::endl;
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  calculate_diagonal(diagonal);

  invert_diagonal(diagonal);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::initialize_dof_vector(VectorType & vector) const
{
  initialize_dof_vector_pressure(vector);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::initialize_dof_vector_pressure(VectorType & vector) const
{
  data->initialize_dof_vector(vector, operator_data.dof_index_pressure);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::initialize_dof_vector_velocity(VectorType & vector) const
{
  data->initialize_dof_vector(vector, operator_data.dof_index_velocity);
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::apply_inverse_block_diagonal(
  VectorType & /*dst*/,
  VectorType const & /*src*/) const
{
  AssertThrow(false,
              ExcMessage(
                "Block Jacobi preconditioner not implemented for compatible Laplace operator."));
}

template<int dim, typename Number>
void
CompatibleLaplaceOperator<dim, Number>::update_inverse_block_diagonal() const
{
  AssertThrow(false,
              ExcMessage("Function update_inverse_block_diagonal() has not been implemented."));
}

template class CompatibleLaplaceOperator<2, float>;
template class CompatibleLaplaceOperator<2, double>;

template class CompatibleLaplaceOperator<3, float>;
template class CompatibleLaplaceOperator<3, double>;

} // namespace IncNS
