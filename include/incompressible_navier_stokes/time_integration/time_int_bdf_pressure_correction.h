/*
 * time_int_bdf_pressure_correction.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_

#include <deal.II/lac/la_parallel_block_vector.h>

#include "time_int_bdf_navier_stokes.h"

namespace IncNS
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorPressureCorrection;

} // namespace Interface

template<typename Number>
class TimeIntBDFPressureCorrection : public TimeIntBDF<Number>
{
public:
  typedef TimeIntBDF<Number> Base;

  typedef typename Base::VectorType      VectorType;
  typedef typename Base::BlockVectorType BlockVectorType;

  typedef Interface::OperatorBase<Number>               InterfaceBase;
  typedef Interface::OperatorPressureCorrection<Number> InterfacePDE;

  TimeIntBDFPressureCorrection(std::shared_ptr<InterfaceBase> operator_base_in,
                               std::shared_ptr<InterfacePDE>  operator_pressure_correction_in,
                               InputParameters const &        param_in);

  void
  postprocessing_stability_analysis();

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  update_time_integrator_constants();

  void
  allocate_vectors() override;

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  setup_derived() override;

  void
  initialize_pressure_on_boundary();

  void
  do_solve_timestep();

  void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  momentum_step();

  void
  rhs_momentum(VectorType & rhs);

  void
  pressure_step();

  void
  projection_step();

  void
  rhs_projection(VectorType & rhs) const;

  void
  pressure_update();

  void
  calculate_chi(double & chi) const;

  void
  rhs_pressure(VectorType & rhs) const;

  void
  prepare_vectors_for_next_timestep() override;

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

  // incremental formulation of pressure-correction scheme
  unsigned int order_pressure_extrapolation;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_gradient;

  // stores pressure Dirichlet boundary values at previous times
  std::vector<VectorType> pressure_dbc;

  std::vector<Number>       computing_times;
  std::vector<unsigned int> iterations;

  unsigned int N_iter_nonlinear_momentum;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_ \
        */
