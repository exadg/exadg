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

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim>
struct BodyForceData
{
  BodyForceData() : dof_index(0), quad_index(0), pull_back_body_force(false)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<Function<dim>> function;

  bool pull_back_body_force;
};

template<int dim, typename Number>
class BodyForceOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef BodyForceOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

public:
  /*
   * Constructor.
   */
  BodyForceOperator();

  /*
   * Initialization.
   */
  void
  initialize(MatrixFree<dim, Number> const & matrix_free, BodyForceData<dim> const & data);

  static MappingFlags
  get_mapping_flags();

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, VectorType const & src, double const time) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  BodyForceData<dim> data;

  double mutable time;
};

} // namespace Structure
} // namespace ExaDG

#endif
