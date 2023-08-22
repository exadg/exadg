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

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>

namespace ExaDG
{
namespace Structure
{
namespace Interface
{
template<typename Number>
class Operator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  Operator()
  {
  }

  virtual ~Operator()
  {
  }

  virtual void
  initialize_dof_vector(VectorType & src) const = 0;

  virtual void
  prescribe_initial_displacement(VectorType & displacement, double const time) const = 0;

  virtual void
  prescribe_initial_velocity(VectorType & velocity, double const time) const = 0;

  virtual void
  compute_initial_acceleration(VectorType &       acceleration,
                               VectorType const & displacement,
                               double const       time) const = 0;

  virtual void
  evaluate_mass_operator(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  apply_add_damping_operator(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  evaluate_add_boundary_mass_operator(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  set_robin_parameters(std::set<dealii::types::boundary_id> const & boundary_IDs,
                       double const & robin_parameter) const = 0;

  virtual std::tuple<unsigned int, unsigned int>
  solve_nonlinear(VectorType &       sol,
                  VectorType const & rhs,
                  double const       scaling_factor_acceleration,
                  double const       scaling_factor_velocity,
                  double const       time,
                  bool const         update_preconditioner) const = 0;

  virtual void
  rhs(VectorType & dst, double const time) const = 0;

  virtual unsigned int
  solve_linear(VectorType &       sol,
               VectorType const & rhs,
               double const       scaling_factor_acceleration,
               double const       scaling_factor_velocity,
               double const       time,
               bool const         update_preconditioner) const = 0;
};

} // namespace Interface
} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
