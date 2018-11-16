/*
 * operator.h
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_INTERFACE_SPACE_TIME_OPERATOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_INTERFACE_SPACE_TIME_OPERATOR_H_

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace ConvDiff
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

  // explicit time integration: evaluate operator
  virtual void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const = 0;

  // explicit time integration: OIF substepping
  virtual void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    double const       evaluation_time) const = 0;

  // explicit time integration of convective term
  virtual void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time) const = 0;

  // implicit time integration: calculate right-hand side of linear system of equations
  virtual void
  rhs(VectorType & dst, double const evaluation_time = 0.0) const = 0;

  // implicit time integration: solve linear system of equations
  virtual unsigned int
  solve(VectorType &       sol,
        VectorType const & rhs,
        double const       scaling_factor  = -1.0,
        double const       evaluation_time = -1.0) = 0;

  // add mass matrix term to rhs-vector
  virtual void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const = 0;

  // time integration: initialize dof vectors
  virtual void
  initialize_dof_vector(VectorType & src) const = 0;

  // time integration: prescribe initial conditions
  virtual void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const = 0;

  // time step calculation: CFL condition (has to loop over all cells and evaluate quantities
  // related to spatial discretization (which is why this function can be found in this interface
  // class)
  virtual double
  calculate_time_step_cfl(double const time,
                          double const cfl,
                          double const exponent_fe_degree) const = 0;

  // needed time step calculation
  virtual double
  calculate_maximum_velocity(double const time) const = 0;

  // needed time step calculation
  virtual double
  calculate_minimum_element_length() const = 0;

  // needed time step calculation
  virtual unsigned int
  get_polynomial_degree() const = 0;

  // postprocessing
  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) const = 0;
};

template<typename Number>
class OperatorOIF
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OperatorOIF(std::shared_ptr<ConvDiff::Interface::Operator<Number>> operator_in)
    : pde_operator(operator_in)
  {
  }

  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    pde_operator->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(dst,
                                                                                  src,
                                                                                  evaluation_time);
  }

  void
  initialize_dof_vector(VectorType & src) const
  {
    pde_operator->initialize_dof_vector(src);
  }

private:
  std::shared_ptr<ConvDiff::Interface::Operator<Number>> pde_operator;
};

} // namespace Interface

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_INTERFACE_SPACE_TIME_OPERATOR_H_ */
