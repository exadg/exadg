/*
 * operator.h
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_

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

  // needed time step calculation
  virtual double
  calculate_minimum_element_length() const = 0;

  // needed time step calculation
  virtual unsigned int
  get_polynomial_degree() const = 0;

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

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
