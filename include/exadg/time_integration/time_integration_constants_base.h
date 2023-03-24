/*
 * time_integration_constants_base.h
 *
 *  Created on: Mar 24, 2023
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_


namespace ExaDG
{
class TimeIntegratorConstantsBase
{
public:
  TimeIntegratorConstantsBase(unsigned int const order_time_integrator,
                              bool const         start_with_low_order_method);

  virtual ~TimeIntegratorConstantsBase()
  {
  }

  /*
   *  This function updates the time integrator constants in case of constant time step sizes.
   */
  void
  update(unsigned int const current_order)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(current_order <= order && start_with_low_order == true)
    {
      set_constant_time_step(current_order);
    }
    else
    {
      set_constant_time_step(order);
    }
  }

  /*
   *  This function updates the time integrator constants in case of adaptive time step sizes.
   */
  void
  update(unsigned int const current_order, std::vector<double> const & time_steps)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(current_order <= order && start_with_low_order == true)
    {
      set_adaptive_time_step(current_order, time_steps);
    }
    else // adjust time integrator constants since this is adaptive time stepping
    {
      set_adaptive_time_step(order, time_steps);
    }
  }

  /*
   *  This function prints the time integrator constants
   */
  virtual void
  print(dealii::ConditionalOStream & pcout) const = 0;


private:
  /*
   *  This function calculates the time integrator constants in case of constant time step sizes.
   */
  virtual void
  set_constant_time_step(unsigned int const current_order) = 0;

  /*
   *  This function calculates time integrator constants in case of varying time step sizes
   * (adaptive time stepping).
   */
  virtual void
  set_adaptive_time_step(unsigned int const          current_order,
                         std::vector<double> const & time_steps) = 0;

  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;
};


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_ */
