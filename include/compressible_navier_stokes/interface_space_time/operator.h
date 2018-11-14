/*
 * operator.h
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_INTERFACE_SPACE_TIME_OPERATOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_INTERFACE_SPACE_TIME_OPERATOR_H_

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace CompNS
{
namespace Interface
{
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

  // postprocessing
  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time,
                    int const          time_step_number) const = 0;

  // analysis of computational costs
  virtual double
  get_wall_time_operator_evaluation() const = 0;
};

} // namespace Interface

} // namespace CompNS


#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_INTERFACE_SPACE_TIME_OPERATOR_H_ */
