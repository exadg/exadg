/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
class InverseMassOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  InverseMassOperator() : matrix_free(nullptr), dof_index(0), quad_index(0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = dof_index_in;
    quad_index        = quad_index_in;
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst.zero_out_ghost_values();

    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &,
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

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;
};

} // namespace ExaDG


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
