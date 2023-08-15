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
#include <numeric>

#include <deal.II/lac/vector.h>

#include <exadg/time_integration/bdf_constants.h>

// Check compute_bdf_time_derivative() from quantities and times

using namespace ExaDG;
using VectorType = dealii::Vector<double>;

VectorType
get_vector_with_value(double const value)
{
  VectorType vec(1);
  vec[0] = value;
  return vec;
}

void
test(unsigned int const order, double const target_slope)
{
  std::cout << "Check derivative of slope " << target_slope << std::endl;

  // construct quantites and times
  std::vector<VectorType> quantities_np_values(order + 1);
  std::vector<double>     times_np(order + 1);

  // current time
  times_np[0] = 10.0;

  // start value
  quantities_np_values[0] = get_vector_with_value(100.0);

  double const dt = 1.0;
  for(unsigned int i = 1; i < order + 1; ++i)
  {
    times_np[i] = times_np[i - 1] - dt;

    quantities_np_values[i] =
      get_vector_with_value(-target_slope * dt + quantities_np_values[i - 1][0]);
  }

  // transfrom to vector of pointers
  std::vector<VectorType const *> quantities_np(quantities_np_values.size());
  std::transform(quantities_np_values.begin(),
                 quantities_np_values.end(),
                 quantities_np.begin(),
                 [](VectorType const & t) { return &t; });

  // compute temporal derivative
  VectorType derivative(1);
  derivative[0] = -99.0;
  compute_bdf_time_derivative(derivative, quantities_np, times_np);

  // check if computed slope equals target slope
  assert(std::abs(derivative[0] - target_slope) < 1e-12);
  std::cout << "OK" << std::endl;
}

int
main()
{
  for(unsigned int order = 1; order <= 4; ++order)
  {
    std::cout << "Order " << order << std::endl;
    test(order, 0.0);
    test(order, -1.0);
  }
  return 0;
}
