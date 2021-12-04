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

#ifndef INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_
#define INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

namespace ExaDG
{
using namespace dealii;

template<int dim>
inline double
calculate_maximum_vertex_distance(typename Triangulation<dim>::active_cell_iterator & cell)
{
  double maximum_vertex_distance = 0.0;

  for(const unsigned int i : cell->vertex_indices())
  {
    Point<dim> & ref_vertex = cell->vertex(i);
    // start the loop with the second vertex!
    for(const unsigned int j : cell->vertex_indices())
    {
      if(j != i)
      {
        Point<dim> & vertex     = cell->vertex(j);
        maximum_vertex_distance = std::max(maximum_vertex_distance, vertex.distance(ref_vertex));
      }
    }
  }

  return maximum_vertex_distance;
}

/*
 * This is a rather naive version of computing the aspect ratio as this version does not
 * detect all modes of deformation (e.g., parallelogram with small angle, assuming that both
 * sides of the parallelogram have the same length, the aspect ratio would tend to a value of
 * 2 if the angle goes to zero!). Instead, the version GridTools::compute_maximum_aspect_ratio()
 * that relies on the computation of singular values of the Jacobian should be used.
 */
template<int dim>
inline double
calculate_aspect_ratio_vertex_distance(Triangulation<dim> const & triangulation,
                                       MPI_Comm const &           mpi_comm)
{
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                    endc = triangulation.end();

  double max_aspect_ratio = 0.0;

  for(; cell != endc; ++cell)
  {
    if(cell->is_locally_owned())
    {
      double minimum_vertex_distance = cell->minimum_vertex_distance();
      double maximum_vertex_distance = calculate_maximum_vertex_distance<dim>(cell);
      // normalize so that a uniform Cartesian mesh has aspect ratio = 1
      double const aspect_ratio =
        (maximum_vertex_distance / minimum_vertex_distance) / std::sqrt(dim);

      max_aspect_ratio = std::max(aspect_ratio, max_aspect_ratio);
    }
  }

  double const global_max_aspect_ratio = Utilities::MPI::max(max_aspect_ratio, mpi_comm);

  return global_max_aspect_ratio;
}

} // namespace ExaDG


#endif /* INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_ */
