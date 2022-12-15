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
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
namespace IP // IP = interior penalty
{
/*
 *  This function calculates the penalty parameter of the interior
 *  penalty method for each cell.
 */
template<int dim, typename Number>
void
calculate_penalty_parameter(
  dealii::AlignedVector<dealii::VectorizedArray<Number>> & array_penalty_parameter,
  dealii::MatrixFree<dim, Number> const &                  matrix_free,
  unsigned int const                                       dof_index = 0)
{
  unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
  array_penalty_parameter.resize(n_cells);

  dealii::Mapping<dim> const &       mapping = *matrix_free.get_mapping_info().mapping;
  dealii::FiniteElement<dim> const & fe      = matrix_free.get_dof_handler(dof_index).get_fe();
  unsigned int const                 degree  = fe.degree;

  dealii::QGauss<dim>   quadrature(degree + 1);
  dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

  dealii::QGauss<dim - 1>   face_quadrature(degree + 1);
  dealii::FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, dealii::update_JxW_values);

  for(unsigned int i = 0; i < n_cells; ++i)
  {
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
    {
      typename dealii::DoFHandler<dim>::cell_iterator cell =
        matrix_free.get_cell_iterator(i, v, dof_index);
      fe_values.reinit(cell);

      // calculate cell volume
      Number volume = 0;
      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        volume += fe_values.JxW(q);
      }

      // calculate surface area
      Number surface_area = 0;
      for(unsigned int const f : cell->face_indices())
      {
        fe_face_values.reinit(cell, f);
        Number const factor = (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) ? 1. : 0.5;
        for(unsigned int q = 0; q < face_quadrature.size(); ++q)
        {
          surface_area += fe_face_values.JxW(q) * factor;
        }
      }

      array_penalty_parameter[i][v] = surface_area / volume;
    }
  }
}

/*
 *  This function returns the penalty factor of the interior penalty method for
 *  quadrilateral/hexahedral or for triangular/tetrahedral elements for a given
 *  polynomial degree of the shape functions, a specified penalty factor
 *  (scaling factor) and the dimension of the problem.
 */

template<int dim, typename Number>
Number
get_penalty_factor(unsigned int const degree, bool const use_simplex, Number const factor = 1.0)
{
  // use penalty factor for simplex elements according to Shahbazi (2005)
  if(use_simplex)
    return factor * (degree + 1.0) * (degree + dim) / dim;
  else
    return factor * (degree + 1.0) * (degree + 1.0);
}

} // namespace IP
} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_ */
