/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_INTERFACE_H_

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>

namespace ExaDG
{
namespace Acoustics
{
namespace Interface
{
template<typename Number>
class SpatialOperator
{
public:
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  virtual ~SpatialOperator() = default;


  // time integration: initialize dof vector
  virtual void
  initialize_dof_vector(BlockVectorType & dst) const = 0;

  // time integration: prescribe initial conditions
  virtual void
  prescribe_initial_conditions(BlockVectorType & src, double const evaluation_time) const = 0;

  // time integration: evaluate
  virtual void
  evaluate(BlockVectorType & dst, BlockVectorType const & src, double const time) const = 0;

  virtual double
  calculate_time_step_cfl() const = 0;
};

} // namespace Interface
} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
