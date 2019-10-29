/*
 * inverse_mass_matrix.h
 *
 *  Created on: May 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

template<int dim, int n_components, typename Number>
class InverseMassMatrixOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassMatrixOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  typedef std::pair<unsigned int, unsigned int> Range;

  InverseMassMatrixOperator() : matrix_free(nullptr), dof_index(0), quad_index(0)
  {
  }

  virtual ~InverseMassMatrixOperator(){};

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              degree_in,
             unsigned int const              dof_index_in,
             unsigned int const              quad_index_in)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = dof_index_in;
    quad_index        = quad_index_in;

    coefficients.resize(Utilities::pow(degree_in + 1, dim));
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  virtual void
  cell_loop(MatrixFree<dim, Number> const &,
            VectorType &       dst,
            VectorType const & src,
            Range const &      cell_range) const
  {
    fe_eval.reset(new Integrator(*matrix_free, dof_index, quad_index));
    inverse.reset(new CellwiseInverseMass(*fe_eval));

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

  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;

  mutable std::shared_ptr<Integrator> fe_eval;

  mutable AlignedVector<VectorizedArray<Number>> coefficients;

  mutable std::shared_ptr<CellwiseInverseMass> inverse;
};


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
