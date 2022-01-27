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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
class VorticityCalculator
{
private:
  typedef VorticityCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  static unsigned int const number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef CellIntegrator<dim, dim, Number> Integrator;

public:
  VorticityCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in);

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index;
  unsigned int quad_index;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_ \
        */
