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

#ifndef INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_
#define INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_

// deal.II
#include <deal.II/grid/tria.h>

namespace ExaDG
{
/*
 *  This function calculates the characteristic element length h
 *  defined as h = min_{e=1,...,N_el} h_e, where h_e is the
 *  minimum vertex distance of element e.
 */
template<int dim>
inline double
calculate_minimum_vertex_distance(dealii::Triangulation<dim> const & triangulation,
                                  MPI_Comm const &                   mpi_comm)
{
  double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();

  for(auto const & cell : triangulation.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      diameter = cell->minimum_vertex_distance();
      if(diameter < min_cell_diameter)
        min_cell_diameter = diameter;
    }
  }

  double const global_min_cell_diameter =
    -dealii::Utilities::MPI::max(-min_cell_diameter, mpi_comm);

  return global_min_cell_diameter;
}

inline double
calculate_characteristic_element_length(double const element_length, unsigned int const fe_degree)
{
  return element_length / ((double)(fe_degree + 1));
}

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_ */
