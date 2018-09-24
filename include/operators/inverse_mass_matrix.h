/*
 * InverseMassMatrix.h
 *
 *  Created on: May 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

using namespace dealii;

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

// Collect all data for the inverse mass matrix operation in a struct in order to avoid allocating
// the memory repeatedly.
template<int dim, int fe_degree, typename Number, int n_components>
struct InverseMassMatrixData
{
  InverseMassMatrixData(MatrixFree<dim, Number> const & data,
                        unsigned int const              fe_index   = 0,
                        unsigned int const              quad_index = 0)
    : fe_eval(1,
              FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>(data,
                                                                                fe_index,
                                                                                quad_index)),
      coefficients(Utilities::pow(fe_degree + 1, dim)),
      inverse(fe_eval[0])
  {
  }

  // Manually implement the copy operator because CellwiseInverseMassMatrix must point to the object
  // 'fe_eval'
  InverseMassMatrixData(InverseMassMatrixData const & other)
    : fe_eval(other.fe_eval), coefficients(other.coefficients), inverse(fe_eval[0])
  {
  }

  // For memory alignment reasons, need to place the FEEvaluation object into an aligned vector
  AlignedVector<FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>> fe_eval;

  AlignedVector<VectorizedArray<Number>> coefficients;

  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, n_components, Number> inverse;
};

template<int dim, int fe_degree, typename value_type, int n_components = dim>
class InverseMassMatrixOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  InverseMassMatrixOperator() : matrix_free_data(nullptr)
  {
  }

  virtual ~InverseMassMatrixOperator(){};

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             unsigned int const                  dof_index,
             unsigned int const                  quad_index)
  {
    this->matrix_free_data = &mf_data;

    // generate initial mass matrix data to avoid allocating it over and over again
    mass_matrix_data.reset(new Threads::ThreadLocalStorage<
                           InverseMassMatrixData<dim, fe_degree, value_type, n_components>>(
      InverseMassMatrixData<dim, fe_degree, value_type, n_components>(*matrix_free_data,
                                                                      dof_index,
                                                                      quad_index)));
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free_data->cell_loop(
      &InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>::local_apply,
      this,
      dst,
      src);
  }

protected:
  MatrixFree<dim, value_type> const * matrix_free_data;
  mutable std::shared_ptr<
    Threads::ThreadLocalStorage<InverseMassMatrixData<dim, fe_degree, value_type, n_components>>>
    mass_matrix_data;

private:
  virtual void
  local_apply(MatrixFree<dim, value_type> const &,
              VectorType &                                  dst,
              VectorType const &                            src,
              std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    InverseMassMatrixData<dim, fe_degree, value_type, n_components> & mass_data =
      mass_matrix_data->get();

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      mass_data.fe_eval[0].reinit(cell);
      mass_data.fe_eval[0].read_dof_values(src, 0);

      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
      mass_data.inverse.apply(mass_data.coefficients,
                              n_components,
                              mass_data.fe_eval[0].begin_dof_values(),
                              mass_data.fe_eval[0].begin_dof_values());

      mass_data.fe_eval[0].set_dof_values(dst, 0);
    }
  }
};


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
