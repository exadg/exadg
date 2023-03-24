/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_BDF_CONSTANTS_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_BDF_CONSTANTS_H_

// C/C++
#include <algorithm>
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/time_integration/time_integration_constants_base.h>

namespace ExaDG
{
/**
 * Class that manages Backward Differentiation Formula time integrator constants.
 */
class BDFTimeIntegratorConstants : public TimeIntegratorConstantsBase
{
public:
  BDFTimeIntegratorConstants(unsigned int const order, bool const start_with_low_order);

  double
  get_gamma0() const;

  double
  get_alpha(unsigned int const i) const;

  void
  print(dealii::ConditionalOStream & pcout) const final;

private:
  void
  set_constant_time_step(unsigned int const current_order) final;

  void
  set_adaptive_time_step(unsigned int const          current_order,
                         std::vector<double> const & time_steps) final;

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
compute_bdf_time_derivative(VectorType &                            derivative,
                            VectorType const &                      solution_np,
                            std::vector<VectorType const *> const & previous_solutions,
                            BDFTimeIntegratorConstants const &      bdf,
                            double const &                          time_step_size)
{
  derivative.equ(bdf.get_gamma0() / time_step_size, solution_np);

  for(unsigned int i = 0; i < previous_solutions.size(); ++i)
    derivative.add(-bdf.get_alpha(i) / time_step_size, *previous_solutions[i]);
}

template<typename VectorType>
void
compute_bdf_time_derivative(VectorType &                       derivative,
                            VectorType const &                 solution_np,
                            std::vector<VectorType> const &    previous_solutions,
                            BDFTimeIntegratorConstants const & bdf,
                            double const &                     time_step_size)
{
  std::vector<VectorType const *> previous_solutions_ptrs(previous_solutions.size());

  std::transform(previous_solutions.begin(),
                 previous_solutions.end(),
                 previous_solutions_ptrs.begin(),
                 [](VectorType const & t) { return &t; });

  compute_bdf_time_derivative(
    derivative, solution_np, previous_solutions_ptrs, bdf, time_step_size);
}

template<typename VectorType>
void
compute_bdf_time_derivative(VectorType &                            derivative,
                            std::vector<VectorType const *> const & previous_solutions_np,
                            std::vector<double> const &             times_np)
{
  AssertThrow(
    previous_solutions_np.size() == times_np.size() and times_np.size() > 0,
    dealii::ExcMessage(
      "times and previous_solutions_np handed to compute_bdf_time_derivative() have different sizes or size 0."));

  std::vector<VectorType const *> previous_solutions(previous_solutions_np.begin() + 1,
                                                     previous_solutions_np.end());

  // compute time step sizes from times
  std::vector<double> time_steps(times_np.size() - 1);
  for(unsigned int i = 0; i < time_steps.size(); ++i)
    time_steps[i] = times_np[i] - times_np[i + 1];


  // construct BDF constants
  unsigned int const         order = previous_solutions.size();
  BDFTimeIntegratorConstants bdf(order, true);
  bdf.update(order, true, time_steps);

  // compute temporal derivative
  compute_bdf_time_derivative(
    derivative, *previous_solutions_np[0], previous_solutions, bdf, time_steps[0]);
}


} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_BDF_CONSTANTS_H_ */
