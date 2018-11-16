/*
 * time_int_bdf_pressure_correction.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "time_int_bdf_navier_stokes.h"

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
class OperatorPressureCorrection;

} // namespace Interface

template<int dim, typename Number>
class TimeIntBDFPressureCorrection : public TimeIntBDF<dim, Number>
{
public:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef Interface::OperatorBase<Number>               InterfaceBase;
  typedef Interface::OperatorPressureCorrection<Number> InterfacePDE;

  TimeIntBDFPressureCorrection(std::shared_ptr<InterfaceBase> operator_base_in,
                               std::shared_ptr<InterfacePDE>  operator_pressure_correction_in,
                               InputParameters<dim> const &   param_in,
                               unsigned int const             n_refine_time_in,
                               bool const                     use_adaptive_time_stepping_in);

  void
  analyze_computing_times() const;

  void
  postprocessing_stability_analysis();

private:
  void
  update_time_integrator_constants();

  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  setup_derived();

  void
  initialize_vec_convective_term();

  void
  initialize_vec_pressure_gradient_term();

  void
  solve_timestep();

  void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  momentum_step();

  void
  rhs_momentum();

  void
  pressure_step();

  void
  projection_step();

  void
  rhs_projection();

  void
  pressure_update();

  void
  calculate_chi(double & chi) const;

  void
  rhs_pressure();

  void
  prepare_vectors_for_next_timestep();

  void
  postprocessing() const;

  void
  postprocessing_steady_problem() const;

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

  VectorType              velocity_np;
  std::vector<VectorType> velocity;

  VectorType              pressure_np;
  std::vector<VectorType> pressure;

  VectorType pressure_increment;

  std::vector<VectorType> vec_convective_term;

  // rhs vector momentum step
  VectorType rhs_vec_momentum;

  // rhs vector pressur step
  VectorType rhs_vec_pressure;
  VectorType rhs_vec_pressure_temp;

  // rhs vector projection step
  VectorType rhs_vec_projection;
  VectorType rhs_vec_projection_temp;

  // incremental formulation of pressure-correction scheme
  unsigned int order_pressure_extrapolation;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_gradient;

  std::vector<VectorType> vec_pressure_gradient_term;

  std::vector<Number>       computing_times;
  std::vector<unsigned int> iterations;

  unsigned int N_iter_nonlinear_momentum;

  // temporary vectors needed for pseudo-time-stepping algorithm
  VectorType velocity_tmp;
  VectorType pressure_tmp;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_ \
        */
