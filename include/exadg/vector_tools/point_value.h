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

#ifndef INCLUDE_VECTOR_TOOLS_POINT_VALUE_H_
#define INCLUDE_VECTOR_TOOLS_POINT_VALUE_H_

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
void
my_point_value(Vector<Number> &                                       result,
               Mapping<dim> const &                                   mapping,
               DoFHandler<dim> const &                                dof_handler,
               LinearAlgebra::distributed::Vector<Number> const &     dof_vector,
               typename DoFHandler<dim>::active_cell_iterator const & cell,
               Point<dim> const &                                     point_in_ref_coord)
{
  Assert(GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord) < 1e-10, ExcInternalError());

  Quadrature<dim> const quadrature(GeometryInfo<dim>::project_to_unit_cell(point_in_ref_coord));

  FiniteElement<dim> const & fe = dof_handler.get_fe();
  FEValues<dim>              fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell);

  // then use this to get the values of the given fe_function at this point
  std::vector<Vector<Number>> solution_vector(1, Vector<Number>(fe.n_components()));
  fe_values.get_function_values(dof_vector, solution_vector);
  result = solution_vector[0];
}

template<int dim, typename Number>
void
evaluate_scalar_quantity_in_point(
  Number &                                           solution_value,
  DoFHandler<dim> const &                            dof_handler,
  Mapping<dim> const &                               mapping,
  LinearAlgebra::distributed::Vector<Number> const & numerical_solution,
  Point<dim> const &                                 point,
  MPI_Comm const &                                   mpi_comm,
  double const                                       tolerance = 1.e-10)
{
  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> Pair;

  std::vector<Pair> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point, tolerance);

  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;
  solution_value       = 0.0;

  // loop over all adjacent cells
  for(auto cell : adjacent_cells)
  {
    // go on only if cell is owned by the processor
    if(cell.first->is_locally_owned())
    {
      Vector<Number> value(1);
      my_point_value(value, mapping, dof_handler, numerical_solution, cell.first, cell.second);

      solution_value += value(0);
      ++counter;
    }
  }

  // parallel computations: add results of all processors and calculate mean value
  counter = Utilities::MPI::sum(counter, mpi_comm);
  AssertThrow(counter > 0, ExcMessage("No points found."));

  solution_value = Utilities::MPI::sum(solution_value, mpi_comm);
  solution_value /= (double)counter;
}

template<int dim, typename Number>
void evaluate_vectorial_quantity_in_point(
  Tensor<1, dim, Number> &                           solution_value,
  DoFHandler<dim> const &                            dof_handler,
  Mapping<dim> const &                               mapping,
  LinearAlgebra::distributed::Vector<Number> const & numerical_solution,
  Point<dim> const &                                 point,
  MPI_Comm const &                                   mpi_comm,
  double const                                       tolerance = 1.e-10)
{
  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> Pair;

  std::vector<Pair> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point, tolerance);

  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;
  solution_value       = 0.0;

  // loop over all adjacent cells
  for(auto cell : adjacent_cells)
  {
    // go on only if cell is owned by the processor
    if(cell.first->is_locally_owned())
    {
      Vector<Number> value(dim);
      my_point_value(value, mapping, dof_handler, numerical_solution, cell.first, cell.second);

      for(unsigned int d = 0; d < dim; ++d)
        solution_value[d] += value(d);

      ++counter;
    }
  }

  // parallel computations: add results of all processors and calculate mean value
  counter = Utilities::MPI::sum(counter, mpi_comm);
  AssertThrow(counter > 0, ExcMessage("No points found."));

  for(unsigned int d = 0; d < dim; ++d)
    solution_value[d] = Utilities::MPI::sum(solution_value[d], mpi_comm);
  solution_value /= (double)counter;
}

} // namespace ExaDG

#endif /* INCLUDE_VECTOR_TOOLS_POINT_VALUE_H_ */
