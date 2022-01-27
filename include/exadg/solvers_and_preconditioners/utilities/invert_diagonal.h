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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_

#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/*
 *  This function inverts the diagonal (element by element).
 *  If diagonal values are very small, the inverse of this
 *  diagonal value is set to 1.0.
 */
template<typename Number>
void
invert_diagonal(dealii::LinearAlgebra::distributed::Vector<Number> & diagonal)
{
  for(unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
  {
    if(std::abs(diagonal.local_element(i)) > 1.0e-10)
      diagonal.local_element(i) = 1.0 / diagonal.local_element(i);
    else
      diagonal.local_element(i) = 1.0;
  }
}

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_ */
