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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_AB_CONSTANTS_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_AB_CONSTANTS_H_

// C/C++
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/time_integration/time_integration_constants_base.h>

namespace ExaDG
{
/**
 * Class that manages Adams--Bashforth time integrator constants.
 */
class ABTimeIntegratorConstants : public TimeIntegratorConstantsBase
{
public:
  ABTimeIntegratorConstants(unsigned int const order, bool const start_with_low_order);

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
   *  AB time integrator constants:
   *
   *  du/dt = (alpha_0 f^{n+1} + alpha_1 f^{n} + alpha_2 f^{n-1} +...)
   */
  std::vector<double> alpha;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_AB_CONSTANTS_H_ */
