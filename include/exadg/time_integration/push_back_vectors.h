/*
 * push_back_vectors.h
 *
 *  Created on: Aug 23, 2016
 *      Author: fehn
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
