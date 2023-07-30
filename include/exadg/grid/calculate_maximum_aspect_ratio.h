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
/*
 * This is a rather naive version of computing the aspect ratio as this version does not
 * detect all modes of deformation (e.g., parallelogram with small angle, assuming that both
 * sides of the parallelogram have the same length, the aspect ratio would tend to a value of
 * 2 if the angle goes to zero!). Instead, the version
 * dealii::GridTools::compute_maximum_aspect_ratio() that relies on the computation of singular
 * values of the Jacobian should be used.
 */
template<int dim>
inline double
calculate_aspect_ratio_vertex_distance(dealii::Triangulation<dim> const & triangulation,
                                       dealii::Mapping<dim> const &       mapping,
                                       MPI_Comm const &                   mpi_comm)
{
  double max_aspect_ratio = 0.0;

  for(auto const & cell : triangulation.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      auto const vertices                = mapping.get_vertices(cell);
      double     minimum_vertex_distance = std::numeric_limits<double>::max();
      double     maximum_vertex_distance = 0;
      for(unsigned int i = 0; i < vertices.size(); ++i)
        for(unsigned int j = i + 1; j < vertices.size(); ++j)
        {
          double const distance   = vertices[i].distance_square(vertices[j]);
          minimum_vertex_distance = std::min(minimum_vertex_distance, distance);
          maximum_vertex_distance = std::max(maximum_vertex_distance, distance);
        }

      // normalize so that a uniform Cartesian mesh has aspect ratio = 1 and take
      // square root to get the actual distances
      double const aspect_ratio =
        std::sqrt(maximum_vertex_distance / (minimum_vertex_distance * dim));

      max_aspect_ratio = std::max(aspect_ratio, max_aspect_ratio);
    }
  }

  return dealii::Utilities::MPI::max(max_aspect_ratio, mpi_comm);
}

} // namespace ExaDG


#endif /* INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_ */
