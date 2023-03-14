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

#ifndef INCLUDE_FUNCTION_WITH_NORMAL_H_
#define INCLUDE_FUNCTION_WITH_NORMAL_H_

namespace ExaDG
{
/*
 * Class that extends the Function class of deal.II by the possibility of using normal vectors.
 */

template<int dim>
class FunctionWithNormal : public dealii::Function<dim>
{
public:
  FunctionWithNormal(unsigned int const n_components, double const time)
    : dealii::Function<dim>(n_components, time)
  {
  }

  virtual ~FunctionWithNormal()
  {
  }

  void
  set_normal_vector(dealii::Tensor<1, dim> normal_vector_in)
  {
    normal_vector = normal_vector_in;
  }

  dealii::Tensor<1, dim>
  get_normal_vector() const
  {
    return normal_vector;
  }

private:
  dealii::Tensor<1, dim> normal_vector;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTION_WITH_NORMAL_H_ */
