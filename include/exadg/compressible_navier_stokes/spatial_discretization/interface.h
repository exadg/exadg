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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
namespace CompNS
{
namespace Interface
{
using namespace dealii;

template<typename Number>
class Operator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  Operator()
  {
  }

  virtual ~Operator()
  {
  }

  // time integration: initialize dof vectors
  virtual void
  initialize_dof_vector(VectorType & src) const = 0;

  // time integration: prescribe initial conditions
  virtual void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const = 0;

  // time step calculation: CFL condition
  virtual double
  calculate_time_step_cfl_global() const = 0;

  // Calculate time step size according to diffusion term
  virtual double
  calculate_time_step_diffusion() const = 0;

  // explicit time integration: evaluate operator
  virtual void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const = 0;

  // analysis of computational costs
  virtual double
  get_wall_time_operator_evaluation() const = 0;
};

} // namespace Interface
} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
