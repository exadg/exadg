/*
 * inverse_mass_matrix.h
 *
 *  Created on: May 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

template<int dim, typename Number>
class InverseMassInterface
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~InverseMassInterface()
  {
  }

  virtual void
  initialize(MatrixFree<dim, Number> const & mf_data,
             unsigned int const              dof_index,
             unsigned int const              quad_index) = 0;

  virtual void
  apply(VectorType & dst, VectorType const & src) const = 0;
};

template<int dim, int degree, typename Number, int n_components = dim>
class InverseMassMatrixOperator : public InverseMassInterface<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  InverseMassMatrixOperator()
    : matrix_free_data(nullptr), coefficients(Utilities::pow(degree + 1, dim))
  {
  }

  virtual ~InverseMassMatrixOperator(){};

  void
  initialize(MatrixFree<dim, Number> const & matrix_free,
             unsigned int const              dof_index,
             unsigned int const              quad_index)
  {
    this->matrix_free_data = &matrix_free;

    fe_eval.reset(new FEEvaluation<dim, degree, degree + 1, n_components, Number>(*matrix_free_data,
                                                                                  dof_index,
                                                                                  quad_index));
    inverse.reset(
      new MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_components, Number>(
        *fe_eval));
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free_data->cell_loop(
      &InverseMassMatrixOperator<dim, degree, Number, n_components>::local_apply, this, dst, src);
  }

private:
  virtual void
  local_apply(MatrixFree<dim, Number> const &,
              VectorType &                                  dst,
              VectorType const &                            src,
              std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval->reinit(cell);
      fe_eval->read_dof_values(src, 0);

      inverse->fill_inverse_JxW_values(coefficients);
      inverse->apply(coefficients,
                     n_components,
                     fe_eval->begin_dof_values(),
                     fe_eval->begin_dof_values());

      fe_eval->set_dof_values(dst, 0);
    }
  }

  MatrixFree<dim, Number> const * matrix_free_data;

  std::shared_ptr<FEEvaluation<dim, degree, degree + 1, n_components, Number>> fe_eval;

  mutable AlignedVector<VectorizedArray<Number>> coefficients;

  std::shared_ptr<MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_components, Number>>
    inverse;
};


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
