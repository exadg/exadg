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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_INTERPOLATE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_INTERPOLATE_H_

namespace ExaDG
{
/*
 *   time t
 *  -------->   t_{n-2}   t_{n-1}   t_{n}   t    t_{n+1}
 *  _______________|_________|________|_____|______|___________\
 *                 |         |        |     |      |           /
 *               sol[2]    sol[1]   sol[0] dst
 */
template<typename VectorType>
void
interpolate(VectorType &                            dst,
            double const &                          time,
            std::vector<VectorType const *> const & solutions,
            std::vector<double> const &             times)
{
  dst = 0;

  // loop over all interpolation points
  for(unsigned int k = 0; k < solutions.size(); ++k)
  {
    // evaluate Lagrange polynomial l_k
    double l_k = 1.0;

    for(unsigned int j = 0; j < solutions.size(); ++j)
    {
      if(j != k)
      {
        l_k *= (time - times[j]) / (times[k] - times[j]);
      }
    }

    dst.add(l_k, *solutions[k]);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_INTERPOLATE_H_ */
