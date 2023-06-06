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

#include <exadg/time_integration/bdf_constants.h>

// Check BDF time integration constants

using namespace ExaDG;

void
test(unsigned int const order)
{
  BDFTimeIntegratorConstants constants(order, true);

  std::cout << "BDFTimeIntegratorConstants of oder " << order << std::endl;
  for(unsigned int current_order = 1; current_order <= order; ++current_order)
  {
    std::cout << "Current order " << current_order << std::endl;
    constants.update(current_order, false, {});

    double sum = constants.get_gamma0();
    for(unsigned int i = 0; i < order; ++i)
      sum -= constants.get_alpha(i);

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
