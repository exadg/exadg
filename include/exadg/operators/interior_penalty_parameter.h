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

#ifndef INCLUDE_EXADG_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_
#define INCLUDE_EXADG_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace IP // IP = interior penalty
{
/*
 *  This function calculates the penalty parameter of the interior penalty
 *  method for each cell as the ratio between the surface area and the volume
 *  of the cell. The algorithm separately runs through cells and faces with
 *  CellIntegrator and FaceIntegrator objects to compute the respective values
 *  and finally combines the result.
 */
template<int dim, typename Number>
void
calculate_penalty_parameter(
  dealii::AlignedVector<dealii::VectorizedArray<Number>> & array_penalty_parameter,
  dealii::MatrixFree<dim, Number> const &                  matrix_free,
  unsigned int const                                       dof_index = 0)
{
  // Start by first computing the surface areas of the cells. Since
  // FaceIntegrator runs through all faces independently of cells, we need to
  // add the values computed on faces to the respective one cell (boundary) or
  // two cells (interior face). In the parallel case, we need to exchange
  // information between the processes as each face is processed by exactly
  // one process, but the area needs to be communicated to two cells. We do
  // this by setting up a parallel vector using the partitioner of the cells
  // in the mesh provided by deal.II's parallel triangulation classes (or
  // simply all cells in serial) and running compress() +
  // update_ghost_values() to communicate the information.
  unsigned int const                         level = matrix_free.get_mg_level();
  LinearAlgebra::distributed::Vector<Number> surface_areas;
  if(auto * tria = dynamic_cast<parallel::TriangulationBase<dim> const *>(
       &matrix_free.get_dof_handler(dof_index).get_triangulation()))
  {
    if(level == numbers::invalid_unsigned_int)
      surface_areas.reinit(tria->global_active_cell_index_partitioner().lock());
    else
      surface_areas.reinit(tria->global_level_cell_index_partitioner(level).lock());
  }
  else
  {
    surface_areas.reinit(
      matrix_free.get_dof_handler(dof_index).get_triangulation().n_active_cells());
  }

  // Lambda function to simplify access to cell index for the two cases of
  // multigrid cells versus active cells
  auto const get_cell_index = [level](typename Triangulation<dim>::cell_iterator const & cell) {
    if(level == numbers::invalid_unsigned_int)
      return cell->global_active_cell_index();
    else
      return cell->global_level_cell_index();
  };

  {
    FaceIntegrator<dim, 1, Number> face_integrator(matrix_free, true, dof_index);

    unsigned int const n_face_batches =
      matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
    for(unsigned int face = 0; face < n_face_batches; ++face)
    {
      face_integrator.reinit(face);
      VectorizedArray<Number> face_area = 0.0;
      for(unsigned int q = 0; q < face_integrator.n_q_points; ++q)
        face_area += face_integrator.JxW(q);

      bool const is_interior_face = face < matrix_free.n_inner_face_batches();

      // For interior faces, the face area is weighted by a factor of 0.5
      // according to the formula by "K. Hillewaert, Development of the
      // Discontinuous Galerkin Method for High-Resolution, Large Scale CFD
      // and Acoustics in Industrial Geometries, Ph.D. thesis, Univ. de
      // Louvain, 2013."
      if(is_interior_face)
        face_area = face_area * 0.5;

      // Write the result to the vector entry corresponding to the respective
      // cell slot. For boundary faces, we write it to only one cell,
      // otherwise to both adjacent cells of the face.
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_face_batch(face); ++v)
      {
        surface_areas(get_cell_index(matrix_free.get_face_iterator(face, v, true).first)) +=
          face_area[v];

        if(is_interior_face)
          surface_areas(get_cell_index(matrix_free.get_face_iterator(face, v, false).first)) +=
            face_area[v];
      }
    }
  }

  surface_areas.compress(VectorOperation::add);
  surface_areas.update_ghost_values();

  // As a second step, we use a CellInterator to compute the volume and then
  // combine it with the surface areas computed before
  unsigned int const n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
  array_penalty_parameter.resize(n_cells);
  {
    CellIntegrator<dim, 1, Number> integrator(matrix_free, dof_index);
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);
      VectorizedArray<Number> volume = 0.0;
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        volume += integrator.JxW(q);

      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        array_penalty_parameter[cell][v] =
          surface_areas(get_cell_index(matrix_free.get_cell_iterator(cell, v))) / volume[v];
    }
  }
}

/*
 *  This function returns the penalty factor of the interior penalty method
 *  for quadrilateral/hexahedral elements for a given polynomial degree of
 *  the shape functions and a specified penalty factor (scaling factor).
 */
template<typename Number>
Number
get_penalty_factor(unsigned int const degree, Number const factor = 1.0)
{
  return factor * (degree + 1.0) * (degree + 1.0);
}

} // namespace IP
} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_ */
