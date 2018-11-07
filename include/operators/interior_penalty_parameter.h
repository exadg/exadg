/*
 * interior_penalty_parameter.h
 *
 *  Created on: Mar 13, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_
#define INCLUDE_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace IP // IP = interior penalty
{
using namespace dealii;

/*
 *  This function calculates the penalty parameter of the interior
 *  penalty method for each cell.
 */
template<int dim, int fe_degree, typename Number>
void
calculate_penalty_parameter(AlignedVector<VectorizedArray<Number>> & array_penalty_parameter,
                            MatrixFree<dim, Number> const &          data,
                            Mapping<dim> const &                     mapping,
                            unsigned int const                       dof_index = 0)
{
  unsigned int n_cells = data.n_cell_batches() + data.n_ghost_cell_batches();
  array_penalty_parameter.resize(n_cells);

  QGauss<dim>   quadrature(fe_degree + 1);
  FEValues<dim> fe_values(mapping,
                          data.get_dof_handler(dof_index).get_fe(),
                          quadrature,
                          update_JxW_values);

  QGauss<dim - 1>   face_quadrature(fe_degree + 1);
  FEFaceValues<dim> fe_face_values(mapping,
                                   data.get_dof_handler(dof_index).get_fe(),
                                   face_quadrature,
                                   update_JxW_values);

  for(unsigned int i = 0; i < n_cells; ++i)
  {
    for(unsigned int v = 0; v < data.n_components_filled(i); ++v)
    {
      typename DoFHandler<dim>::cell_iterator cell = data.get_cell_iterator(i, v, dof_index);
      fe_values.reinit(cell);

      // calculate cell volume
      Number volume = 0;
      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        volume += fe_values.JxW(q);
      }

      // calculate surface area
      Number surface_area = 0;
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
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
 *  This function returns the penalty factor of the interior penalty method
 *  for quadrilateral/hexahedral elements for a given polynomial degree of
 *  the shape functions and a specified penalty factor (scaling factor).
 */

template<typename Number>
Number
get_penalty_factor(unsigned int const fe_degree, Number const factor = 1.0)
{
  return factor * (fe_degree + 1.0) * (fe_degree + 1.0);
}

} // namespace IP


#endif /* INCLUDE_OPERATORS_INTERIOR_PENALTY_PARAMETER_H_ */
