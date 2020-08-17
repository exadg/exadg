/*
 * inverse_mass_matrix.h
 *
 *  Created on: May 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/operators.h>
#include "../matrix_free/integrators.h"

namespace ExaDG
{
using namespace dealii;

template<int dim, int n_components, typename Number>
class InverseMassMatrixOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassMatrixOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  InverseMassMatrixOperator() : matrix_free(nullptr), dof_index(0), quad_index(0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              dof_index_in,
             unsigned int const              quad_index_in)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = dof_index_in;
    quad_index        = quad_index_in;
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst.zero_out_ghosts();

    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const &,
            VectorType &       dst,
            VectorType const & src,
            Range const &      cell_range) const
  {
    Integrator          integrator(*matrix_free, dof_index, quad_index);
    CellwiseInverseMass inverse(integrator);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src, 0);

      inverse.apply(integrator.begin_dof_values(), integrator.begin_dof_values());

      integrator.set_dof_values(dst, 0);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;
};

} // namespace ExaDG


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
