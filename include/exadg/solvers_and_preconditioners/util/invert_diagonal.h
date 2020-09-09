/*
 * invert_diagonal.h
 *
 *  Created on: Dec 7, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_

#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
using namespace dealii;

/*
 *  This function inverts the diagonal (element by element).
 *  If diagonal values are very small, the inverse of this
 *  diagonal value is set to 1.0.
 */
template<typename Number>
void
invert_diagonal(LinearAlgebra::distributed::Vector<Number> & diagonal)
{
  for(unsigned int i = 0; i < diagonal.local_size(); ++i)
  {
    if(std::abs(diagonal.local_element(i)) > 1.0e-10)
      diagonal.local_element(i) = 1.0 / diagonal.local_element(i);
    else
      diagonal.local_element(i) = 1.0;
  }
}

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_ */
