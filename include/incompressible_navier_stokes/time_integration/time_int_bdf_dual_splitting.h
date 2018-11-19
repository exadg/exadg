/*
 * time_int_bdf_dual_splitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "time_int_bdf_navier_stokes.h"

using namespace dealii;

namespace IncNS
{
// forward declarations
template<int dim>
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorDualSplitting;

} // namespace Interface

template<int dim, typename Number>
class TimeIntBDFDualSplitting : public TimeIntBDF<dim, Number>
{
public:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef Interface::OperatorBase<Number>          InterfaceBase;
  typedef Interface::OperatorDualSplitting<Number> InterfacePDE;

  TimeIntBDFDualSplitting(std::shared_ptr<InterfaceBase> operator_base_in,
                          std::shared_ptr<InterfacePDE>  pde_operator_in,
                          InputParameters<dim> const &   param_in,
                          unsigned int const             n_refine_time_in,
                          bool const                     use_adaptive_time_stepping_in);

  void
  analyze_computing_times() const;

  void
  postprocessing_stability_analysis();

private:
  void
  setup_derived();

  void
  solve_timestep();

  void
  allocate_vectors();

  void
  prepare_vectors_for_next_timestep();

  void
  convective_step();

  void
  postprocessing() const;

  void
  postprocessing_steady_problem() const;

  void
  update_time_integrator_constants();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vorticity();

  void
  initialize_vec_convective_term();

  void
  initialize_intermediate_velocity();

  void
  pressure_step();

  void
  rhs_pressure();

  void
  projection_step();

  void
  rhs_projection();

  void
  viscous_step();

  void
  rhs_viscous();

  void
  solve_steady_problem();

  double
  evaluate_residual();

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  std::shared_ptr<InterfacePDE> pde_operator;

  std::vector<VectorType> velocity;

  std::vector<VectorType> pressure;

  VectorType velocity_np;

  std::vector<VectorType> vorticity;

  std::vector<VectorType> vec_convective_term;

  std::vector<double>       computing_times;
  std::vector<unsigned int> iterations;

  VectorType rhs_vec_viscous;

  // postprocessing: intermediate velocity
  VectorType intermediate_velocity;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;

  VectorType pressure_np;

  VectorType vorticity_extrapolated;

  VectorType rhs_vec_pressure;
  VectorType rhs_vec_pressure_temp;

  VectorType rhs_vec_projection;
  VectorType rhs_vec_projection_temp;

  // temporary vectors needed for pseudo-time-stepping algorithm
  VectorType velocity_tmp;
  VectorType pressure_tmp;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ */
