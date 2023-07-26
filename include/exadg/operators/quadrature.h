/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_OPERATORS_QUADRATURE_H_
#define INCLUDE_EXADG_OPERATORS_QUADRATURE_H_

#include <exadg/grid/grid_data.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
template<int dim>
std::shared_ptr<dealii::Quadrature<dim>>
create_quadrature(ElementType const & element_type, unsigned int const n_q_points_1d)
{
  std::shared_ptr<dealii::Quadrature<dim>> quadrature;
  if(element_type == ElementType::Hypercube)
  {
    quadrature = std::make_shared<dealii::QGauss<dim>>(n_q_points_1d);
  }
  else if(element_type == ElementType::Simplex)
  {
    quadrature = std::make_shared<dealii::QGaussSimplex<dim>>(n_q_points_1d);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  return quadrature;
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_QUADRATURE_H_ */
