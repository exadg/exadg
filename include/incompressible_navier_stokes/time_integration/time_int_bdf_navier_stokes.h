/*
 * time_int_bdf_navier_stokes.h
 *
 *  Created on: Jun 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/time_int_bdf_base.h"

using namespace dealii;

namespace IncNS
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorOIF;

} // namespace Interface


template<typename Number>
class TimeIntBDF : public TimeIntBDFBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef Interface::OperatorBase<Number> InterfaceBase;

  TimeIntBDF(std::shared_ptr<InterfaceBase> operator_in, InputParameters const & param_in);

  virtual ~TimeIntBDF()
  {
  }

  virtual VectorType const &
  get_velocity() const = 0;

  void
  get_velocities_and_times(std::vector<VectorType const *> & velocities,
                           std::vector<double> &             times) const;

  // ALE
  // TODO make this function protected again
  std::vector<double>
  get_current_time_integrator_constants() const;

  void
  set_grid_velocity_cfl(VectorType u_grid_cfl_in);

  virtual void
  update_time_integrator_constants();

protected:
  virtual void
  setup_derived() override;

  void
  solve_timestep();

  bool
  print_solver_info() const;

  /*
   * This function implements the OIF sub-stepping algorithm. Has to be implemented here
   * since functionality is related to incompressible flows only (nonlinear convective term).
   */
  void
  calculate_sum_alphai_ui_oif_substepping(double const cfl, double const cfl_oif);

  InputParameters const & param;

  // BDF time integration: Sum_i (alpha_i/dt * u_i)
  VectorType sum_alphai_ui;

  // global cfl number
  double const cfl;

  // cfl number cfl_oif for operator-integration-factor splitting
  double const cfl_oif;

  std::shared_ptr<InterfaceBase> operator_base;

private:
  virtual void
  do_solve_timestep() = 0;

  void
  ale_update_pre()
  {
    // TODO needs to be filled (should be the same for all Navier-Stokes solvers, so
    // there should be no need to make this function virtual)
  }

  virtual void
  ale_update_post() = 0;

  void
  initialize_oif();

  void
  initialize_solution_oif_substepping(unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  virtual void
  solve_steady_problem() = 0;

  virtual void
  postprocessing_steady_problem() const = 0;

  virtual VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const = 0;

  virtual VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const = 0;

  virtual void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */) = 0;

  virtual void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */) = 0;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  // Operator-integration-factor splitting for convective term
  std::shared_ptr<Interface::OperatorOIF<Number>> convective_operator_OIF;

  // OIF splitting
  std::shared_ptr<ExplicitTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>>
    time_integrator_OIF;

  // solution vectors needed for OIF substepping of convective term
  VectorType solution_tilde_m;
  VectorType solution_tilde_mp;

  // CFL ALE
  VectorType u_grid_cfl;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_ */
