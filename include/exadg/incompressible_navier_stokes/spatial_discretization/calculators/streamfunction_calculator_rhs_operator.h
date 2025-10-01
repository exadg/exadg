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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  This function calculates the right-hand side of the Laplace equation that is solved in order to
 * obtain the streamfunction psi
 *
 *    - laplace(psi) = omega  (where omega is the vorticity).
 *
 *  Note that this function can only be used for the two-dimensional case (dim==2).
 */
template<int dim, typename Number>
class StreamfunctionCalculatorRHSOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef StreamfunctionCalculatorRHSOperator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> IntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   IntegratorScalar;

public:
  StreamfunctionCalculatorRHSOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_u_in,
             unsigned int const                      dof_index_u_scalar_in,
             unsigned int const                      quad_index_in);

  void
  apply(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_ \
        */
