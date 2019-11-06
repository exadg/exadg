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

  virtual double
  calculate_time_step_cfl(VectorType const & velocity,
                          double const       cfl,
                          double const       exponent_degree) const = 0;

  virtual double
  calculate_minimum_element_length() const = 0;

  virtual unsigned int
  get_polynomial_degree() const = 0;

  virtual void
  initialize_vector_velocity(VectorType & src) const = 0;

  virtual void
  initialize_vector_velocity_scalar(VectorType & src) const = 0;

  virtual void
  initialize_vector_pressure(VectorType & src) const = 0;

  virtual void
  prescribe_initial_conditions(VectorType & velocity,
                               VectorType & pressure,
                               double const time) const = 0;

  virtual void
  evaluate_add_body_force_term(VectorType & dst, double const time) const = 0;

  virtual void
  evaluate_convective_term(VectorType & dst, VectorType const & src, Number const time) const = 0;

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

  virtual void
  evaluate_pressure_gradient_term(VectorType &       dst,
                                  VectorType const & src,
                                  double const       time) const = 0;

  virtual void
  evaluate_velocity_divergence_term(VectorType &       dst,
                                    VectorType const & src,
                                    double const       time) const = 0;

  virtual void
  apply_mass_matrix(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  apply_inverse_mass_matrix(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  shift_pressure(VectorType & pressure, double const & time = 0.0) const = 0;

  virtual void
  shift_pressure_mean_value(VectorType & pressure, double const & time = 0.0) const = 0;

  virtual void
  update_turbulence_model(VectorType const & velocity) = 0;

  virtual void
  update_projection_operator(VectorType const & velocity, double const time_step_size) const = 0;

  virtual unsigned int
  solve_projection(VectorType &       dst,
                   VectorType const & src,
                   bool const &       update_preconditioner) const = 0;

  virtual void
  compute_vorticity(VectorType & dst, VectorType const & src) const = 0;

  // Arbitrary Lagrangian-Eulerian (ALE) formulation
  virtual void
  update_after_mesh_movement() = 0;

  virtual void
  move_mesh(double const time) = 0;

  virtual void
  move_mesh_and_fill_grid_coordinates_vector(VectorType & vector, double const time) = 0;

  virtual void
  set_grid_velocity(VectorType velocity) = 0;
};

/*
 * Coupled (monolithic) solution approach.
 */
template<typename Number>
class OperatorCoupled
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  OperatorCoupled()
  {
  }

  virtual ~OperatorCoupled()
  {
  }

  virtual void
  initialize_block_vector_velocity_pressure(BlockVectorType & src) const = 0;

  virtual void
  set_scaling_factor_continuity(double const scaling_factor) = 0;

  virtual void
  update_divergence_penalty_operator(VectorType const & velocity) = 0;

  virtual void
  update_continuity_penalty_operator(VectorType const & velocity) = 0;

  virtual void
  rhs_stokes_problem(BlockVectorType & dst, double const & time = 0.0) const = 0;

  virtual unsigned int
  solve_linear_stokes_problem(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              bool const &            update_preconditioner,
                              double const &          time                            = 0.0,
                              double const &          scaling_factor_mass_matrix_term = 1.0) = 0;

  virtual void
  evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                     BlockVectorType const & src,
                                     double const &          time) const = 0;

  virtual void
  solve_nonlinear_problem(BlockVectorType &  dst,
                          VectorType const & sum_alphai_ui,
                          double const &     time,
                          bool const &       update_preconditioner,
                          double const &     scaling_factor_mass_matrix_term,
                          unsigned int &     newton_iterations,
                          unsigned int &     linear_iterations) = 0;

  virtual void
  solve_nonlinear_steady_problem(BlockVectorType &  dst,
                                 VectorType const & rhs_vector,
                                 bool const &       update_preconditioner,
                                 unsigned int &     newton_iterations,
                                 unsigned int &     linear_iterations) = 0;

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    unsigned int const time_step_number) const = 0;

  virtual void
  do_postprocessing_steady_problem(VectorType const & velocity,
                                   VectorType const & pressure) const = 0;
};

/*
 * Dual splitting scheme (velocity correction scheme).
 */
template<typename Number>
class OperatorDualSplitting
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OperatorDualSplitting()
  {
  }

  virtual ~OperatorDualSplitting()
  {
  }

  virtual void
  evaluate_body_force_and_apply_inverse_mass_matrix(VectorType & dst, double const time) const = 0;

  virtual void
  evaluate_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                         VectorType const & src,
                                                         double const       time) const = 0;

  virtual unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  apply_velocity_divergence_term(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs_velocity_divergence_term(VectorType & dst, double const & time) const = 0;

  virtual void
  rhs_ppe_div_term_convective_term_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs_ppe_div_term_body_forces_add(VectorType & dst, double const & time) = 0;

  virtual void
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const = 0;

  virtual void
  rhs_ppe_nbc_add(VectorType & dst, double const & time) = 0;

  virtual void
  rhs_ppe_viscous_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs_ppe_convective_add(VectorType & dst, VectorType const & src) const = 0;

  virtual unsigned int
  solve_viscous(VectorType &       dst,
                VectorType const & src,
                bool const &       update_preconditioner,
                double const &     scaling_factor_time_derivative_term) = 0;

  virtual void
  rhs_add_viscous_term(VectorType & dst, double const time) const = 0;

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    unsigned int const time_step_number) const = 0;

  virtual void
  do_postprocessing_steady_problem(VectorType const & velocity,
                                   VectorType const & pressure) const = 0;
};

/*
 * Pressure-correction scheme.
 */
template<typename Number>
class OperatorPressureCorrection
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OperatorPressureCorrection()
  {
  }

  virtual ~OperatorPressureCorrection()
  {
  }

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    unsigned int const time_step_number) const = 0;

  virtual void
  do_postprocessing_steady_problem(VectorType const & velocity,
                                   VectorType const & pressure) const = 0;

  virtual void
  solve_linear_momentum_equation(VectorType &       solution,
                                 VectorType const & rhs,
                                 bool const &       update_preconditioner,
                                 double const &     scaling_factor_mass_matrix_term,
                                 unsigned int &     linear_iterations) = 0;

  virtual void
  solve_nonlinear_momentum_equation(VectorType &       dst,
                                    VectorType const & rhs_vector,
                                    double const &     time,
                                    bool const &       update_preconditioner,
                                    double const &     scaling_factor_mass_matrix_term,
                                    unsigned int &     newton_iterations,
                                    unsigned int &     linear_iterations) = 0;

  virtual void
  rhs_add_viscous_term(VectorType & dst, double const time) const = 0;

  virtual unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs_ppe_laplace_add(VectorType & dst, double const & time) const = 0;

  virtual void
  apply_inverse_pressure_mass_matrix(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs_pressure_gradient_term(VectorType & dst, double const time) const = 0;

  virtual void
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     time) const = 0;
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
