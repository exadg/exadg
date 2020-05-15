/*
 * time_int_gen_alpha_base.h
 *
 *  Created on: 20.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_
#define INCLUDE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_


#include "enum_types.h"
#include "time_int_base.h"

template<typename Number>
class TimeIntGenAlphaBase : public TimeIntBase
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  TimeIntGenAlphaBase(double const &       start_time_,
                      double const &       end_time_,
                      unsigned int const   max_number_of_time_steps_,
                      double const         spectral_radius_,
                      GenAlphaType const & gen_alpha_type_,
                      RestartData const &  restart_data_,
                      MPI_Comm const &     mpi_comm_);

  double
  get_time_step_size() const;

  void
  set_current_time_step_size(double const & time_step_size);

protected:
  double
  get_scaling_factor_mass() const;

  double
  get_mid_time() const;

  void
  compute_const_vector(VectorType &       const_vector,
                       VectorType const & displacement_n,
                       VectorType const & velocity_n,
                       VectorType const & acceleration_n) const;

  void
  update_displacement(VectorType & displacement_np, VectorType const & displacement_n) const;

  void
  update_velocity(VectorType &       velocity_np,
                  VectorType const & displacement_np,
                  VectorType const & displacement_n,
                  VectorType const & velocity_n,
                  VectorType const & acceleration_n) const;

  void
  update_acceleration(VectorType &       acceleration_np,
                      VectorType const & displacement_np,
                      VectorType const & displacement_n,
                      VectorType const & velocity_n,
                      VectorType const & acceleration_n) const;

private:
  void
  do_timestep_pre_solve(bool const print_header) override;

  void
  do_timestep_post_solve() override;

  virtual void
  prepare_vectors_for_next_timestep() = 0;

  /*
   * returns whether solver info has to be written in the current time step.
   */
  virtual bool
  print_solver_info() const = 0;

  double spectral_radius;
  double alpha_m;
  double alpha_f;
  double beta;
  double gamma;

  double time_step;
};


#endif /* INCLUDE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_BASE_H_ */
