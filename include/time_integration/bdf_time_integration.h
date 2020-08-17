/*
 * bdf_time_integration.h
 *
 *  Created on: Jan 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_
#define INCLUDE_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_

#include <deal.II/base/conditional_ostream.h>

#include <vector>

namespace ExaDG
{
using namespace dealii;

class BDFTimeIntegratorConstants
{
public:
  BDFTimeIntegratorConstants(unsigned int const order_time_integrator,
                             bool const         start_with_low_order_method);

  double
  get_gamma0() const;

  double
  get_alpha(unsigned int const i) const;

  /*
   *  This function updates the time integrator constants of the BDF scheme
   *  in case of constant time step sizes.
   */
  void
  update(unsigned int const time_step_number);

  /*
   *  This function updates the time integrator constants of the BDF scheme
   *  in case of adaptive time step sizes.
   */
  void
  update(unsigned int const time_step_number, std::vector<double> const & time_steps);

  /*
   *  This function prints the time integrator constants
   */
  void
  print(ConditionalOStream & pcout) const;


private:
  /*
   *  This function calculates the time integrator constants of the BDF scheme
   *  in case of constant time step sizes.
   */
  void
  set_constant_time_step(unsigned int const current_order);

  /*
   *  This function calculates time integrator constants
   *  in case of varying time step sizes (adaptive time stepping).
   */
  void
  set_adaptive_time_step(unsigned int const current_order, std::vector<double> const & time_steps);

  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;

  /*
   *  BDF time integrator constants:
   *
   *  du/dt = (gamma_0 u^{n+1} - alpha_0 u^{n} - alpha_1 u^{n-1} ... - alpha_{J-1} u^{n-J+1})/dt
   */
  double gamma0;

  std::vector<double> alpha;
};

/*
 * Calculates the time derivative
 *
 *  derivative = du/dt = (gamma_0 u^{n+1} - alpha_0 u^{n} - alpha_1 u^{n-1} ... - alpha_{J-1}
 * u^{n-J+1})/dt
 */
template<typename VectorType>
void
compute_bdf_time_derivative(VectorType &                       derivative,
                            VectorType const &                 solution_np,
                            std::vector<VectorType> const &    previous_solutions,
                            BDFTimeIntegratorConstants const & bdf,
                            double const &                     time_step_size)
{
  derivative.equ(bdf.get_gamma0() / time_step_size, solution_np);

  for(unsigned int i = 0; i < previous_solutions.size(); ++i)
    derivative.add(-bdf.get_alpha(i) / time_step_size, previous_solutions[i]);
}

} // namespace ExaDG

#endif /* INCLUDE_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_ */
