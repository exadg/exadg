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

#ifndef INCLUDE_EXADG_GRID_HYPER_RECTANGLE_H_
#define INCLUDE_EXADG_GRID_HYPER_RECTANGLE_H_

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

// ExaDG
#include <exadg/grid/enum_types.h>

namespace ExaDG
{
/**
 * This function wraps around dealii::GridGenerator::subdivided_hyper_rectangle() taking an
 * additional argument ElementType.
 */
template<int dim>
void
create_subdivided_hyper_rectangle(dealii::Triangulation<dim> &      tria,
                                  std::vector<unsigned int> const & repetitions,
                                  dealii::Point<dim> const &        p1,
                                  dealii::Point<dim> const &        p2,
                                  ElementType const &               element_type,
                                  bool const                        colorize = false)
{
  if(element_type == ElementType::Hypercube)
  {
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2, colorize);
  }
  else if(element_type == ElementType::Simplex)
  {
    dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(
      tria, repetitions, p1, p2, colorize);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("The function create_subdivided_hyper_rectangle() currently "
                                   "supports ElementType::Hypercube and ElementType::Simplex."));
  }
}
} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_HYPER_RECTANGLE_H_ */
