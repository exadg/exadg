/*
 * interpolate.h
 *
 *  Created on: Nov 23, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_INTERPOLATE_H_
#define INCLUDE_TIME_INTEGRATION_INTERPOLATE_H_

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

#endif /* INCLUDE_TIME_INTEGRATION_INTERPOLATE_H_ */
