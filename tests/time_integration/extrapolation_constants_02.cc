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

#include <iostream>

#include <exadg/time_integration/extrapolation_constants.h>

// Check extrapolation constants in case of adaptive timestepping

using namespace ExaDG;

void
test(unsigned int const order)
{
  ExtrapolationConstants constants(order, true);
  std::vector<double>    time_steps{0.1, 0.2, 0.3, 0.4, 0.5};

  std::cout << "ExtrapolationConstants of oder " << order << std::endl;
  for(unsigned int current_order = 1; current_order <= order; ++current_order)
  {
    std::cout << "Current order " << current_order << std::endl;
    constants.update(current_order, true, time_steps);

    double sum = 0.0;
    for(unsigned int i = 0; i < order; ++i)
      sum += constants.get_beta(i);

    std::cout << "Sum: " << sum << std::endl;
  }
}


int
main()
{
  test(1);
  test(2);
  test(3);
  test(4);
  return 0;
}
