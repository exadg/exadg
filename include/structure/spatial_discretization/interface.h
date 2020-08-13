/*
 * interface.h
 *
 *  Created on: 02.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>

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
  apply_mass_matrix(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  compute_rhs_linear(VectorType & dst, double const time) const = 0;

  virtual void
  set_constrained_values_to_zero(VectorType & vector) const = 0;

  virtual std::tuple<unsigned int, unsigned int>
  solve_nonlinear(VectorType &       sol,
                  VectorType const & rhs,
                  double const       factor,
                  double const       time,
                  bool const         update_preconditioner) const = 0;

  virtual unsigned int
  solve_linear(VectorType &       sol,
               VectorType const & rhs,
               double const       factor,
               double const       time) const = 0;
};

} // namespace Interface
} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
