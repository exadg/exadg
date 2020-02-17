/*
 * operator.h
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/dofs/dof_handler.h>

#include "time_integration/interpolate.h"

using namespace dealii;

namespace IncNS
{
namespace Interface
{
/*
 * Base operator for incompressible Navier-Stokes solvers.
 */
template<typename Number>
class OperatorBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OperatorBase()
  {
  }

  virtual ~OperatorBase()
  {
  }

  virtual void
  initialize_vector_velocity(VectorType & src) const = 0;

  virtual void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                                  VectorType const & src,
                                                                  Number const time) const = 0;

  virtual void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    Number const       time,
    VectorType const & solution_interpolated) const = 0;
};

/*
 * Operator-integration-factor (OIF) sub-stepping.
 */
template<typename Number>
class OperatorOIF
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OperatorOIF(std::shared_ptr<IncNS::Interface::OperatorBase<Number>> operator_in)
    : pde_operator(operator_in),
      transport_with_interpolated_velocity(true) // TODO adjust this parameter manually
  {
    if(transport_with_interpolated_velocity)
      initialize_dof_vector(solution_interpolated);
  }

  void
  initialize_dof_vector(VectorType & src) const
  {
    pde_operator->initialize_vector_velocity(src);
  }

  // OIF splitting (transport with interpolated velocity)
  void
  set_solutions_and_times(std::vector<VectorType const *> const & solutions_in,
                          std::vector<double> const &             times_in)
  {
    solutions = solutions_in;
    times     = times_in;
  }

  void
  evaluate(VectorType & dst, VectorType const & src, double const time) const
  {
    if(transport_with_interpolated_velocity)
    {
      interpolate(solution_interpolated, time, solutions, times);

      pde_operator->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
        dst, src, time, solution_interpolated);
    }
    else // nonlinear transport (standard convective term)
    {
      pde_operator->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(dst, src, time);
    }
  }

private:
  std::shared_ptr<IncNS::Interface::OperatorBase<Number>> pde_operator;

  // OIF splitting (transport with interpolated velocity)
  bool                            transport_with_interpolated_velocity;
  std::vector<VectorType const *> solutions;
  std::vector<double>             times;
  VectorType mutable solution_interpolated;
};

} // namespace Interface

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
