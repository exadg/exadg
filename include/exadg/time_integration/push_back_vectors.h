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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_PUSH_BACK_VECTORS_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_PUSH_BACK_VECTORS_H_

// C/C++
#include <vector>

namespace ExaDG
{
/*
 * This function implements a push-back operation that is needed in multistep time integration
 * schemes like BDF schemes in order to update the solution vectors from one time step to the
 * next. The prerequisite to call this function is that the type VectorType implements a
 * swap-function!
 */
template<typename VectorType>
void
push_back(std::vector<VectorType> & vector)
{
  /*
   *   time t
   *  -------->   t_{n-3}   t_{n-2}   t_{n-1}   t_{n}
   *  _______________|_________|________|_________|___________\
   *                 |         |        |         |           /
   *
   *  vector:     vec[3]    vec[2]    vec[1]    vec[0]
   *
   * <- vec[3] <- vec[2] <- vec[1] <- vec[0] <- vec[3] <--
   * |___________________________________________________|
   *
   */

  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(int i = vector.size() - 1; i > 0; --i)
  {
    vector[i].swap(vector[i - 1]);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_PUSH_BACK_VECTORS_H_ */
