/*
 * evaluate_solution_in_given_point.h
 *
 *  Created on: Mar 15, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_
#define INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

template<int dim, typename Number>
void
my_point_value(Vector<Number> &                                       value,
               Mapping<dim> const &                                   mapping,
               DoFHandler<dim> const &                                dof_handler,
               LinearAlgebra::distributed::Vector<Number> const &     solution,
               typename DoFHandler<dim>::active_cell_iterator const & cell,
               Point<dim> const &                                     point_in_ref_coord)
{
  Assert(GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord) < 1e-10, ExcInternalError());

  const FiniteElement<dim> & fe = dof_handler.get_fe();

  const Quadrature<dim> quadrature(GeometryInfo<dim>::project_to_unit_cell(point_in_ref_coord));

  FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell);

  // then use this to get the values of the given fe_function at this point
  std::vector<Vector<Number>> solution_value(1, Vector<Number>(fe.n_components()));
  fe_values.get_function_values(solution, solution_value);
  value = solution_value[0];
}

template<int dim, typename Number>
void
evaluate_scalar_quantity_in_point(
  Number &                                           solution_value,
  DoFHandler<dim> const &                            dof_handler,
  Mapping<dim> const &                               mapping,
  LinearAlgebra::distributed::Vector<Number> const & numerical_solution,
  Point<dim> const &                                 point,
  MPI_Comm const &                                   mpi_comm)
{
  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;

  solution_value = 0.0;

  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> MY_PAIR;

  std::vector<MY_PAIR> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point, 1.e-10);

  // loop over all adjacent cells
  for(typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin();
      cell != adjacent_cells.end();
      ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
      Vector<Number> value(1);
      my_point_value(value, mapping, dof_handler, numerical_solution, cell->first, cell->second);

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
  MPI_Comm const &                                   mpi_comm)
{
  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;

  solution_value = 0.0;

  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> MY_PAIR;

  std::vector<MY_PAIR> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point);

  // loop over all adjacent cells
  for(typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin();
      cell != adjacent_cells.end();
      ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
      Vector<Number> value(dim);
      my_point_value(value, mapping, dof_handler, numerical_solution, cell->first, cell->second);

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

/*
 *  For a given point in physical space, find all adjacent cells and store the global dof index of
 * the first dof of the cell as well as the shape function values of all dofs (to be used for
 * interpolation of the solution in the given point afterwards). (global_dof_index, shape_values)
 * are stored in a vector where each entry corresponds to one adjacent, locally-owned cell.
 */
template<int dim, typename Number>
void
get_global_dof_index_and_shape_values(
  DoFHandler<dim> const &                                     dof_handler,
  Mapping<dim> const &                                        mapping,
  LinearAlgebra::distributed::Vector<Number> const &          numerical_solution,
  Point<dim> const &                                          point,
  std::vector<std::pair<unsigned int, std::vector<Number>>> & global_dof_index_and_shape_values)
{
  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator /*cell*/,
                    Point<dim> /*in ref-coordinates*/>
    MY_PAIR;

  std::vector<MY_PAIR> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point);

  // loop over all adjacent cells
  for(typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin();
      cell != adjacent_cells.end();
      ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
      Assert(GeometryInfo<dim>::distance_to_unit_cell(cell->second) < 1e-10, ExcInternalError());

      const FiniteElement<dim> & fe = dof_handler.get_fe();

      const Quadrature<dim> quadrature(GeometryInfo<dim>::project_to_unit_cell(cell->second));

      FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
      fe_values.reinit(cell->first);
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell->first->get_dof_indices(dof_indices);
      unsigned int global_dof_index =
        numerical_solution.get_partitioner()->global_to_local(dof_indices[0]);
      std::vector<Number> fe_shape_values(fe.dofs_per_cell);
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        fe_shape_values[i] = fe_values.shape_value(i, 0);

      global_dof_index_and_shape_values.emplace_back(global_dof_index, fe_shape_values);
    }
  }
}

/*
 *  Interpolate solution in point by using precomputed shape functions values (for efficiency!)
 *  Noet that we assume that we are dealing with discontinuous finite elements.
 *
 *  The quantity to be evaluated is of type Tensor<1,dim,Number>.
 */
template<int dim, typename Number>
void
interpolate_value_vectorial_quantity(DoFHandler<dim> const &                            dof_handler,
                                     LinearAlgebra::distributed::Vector<Number> const & solution,
                                     unsigned int const &        global_dof_index,
                                     std::vector<Number> const & fe_shape_values,
                                     Tensor<1, dim, Number> &    result)
{
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  Number const * sol_ptr = solution.begin() + global_dof_index;

  for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    result[fe.system_to_component_index(i).first] += sol_ptr[i] * fe_shape_values[i];
}

/*
 *  Interpolate solution in point by using precomputed shape functions values (for efficiency!)
 *  Noet that we assume that we are dealing with discontinuous finite elements.
 *
 *  The quantity to be evaluated is a scalar quantity.
 */
template<int dim, typename Number>
void
interpolate_value_scalar_quantity(DoFHandler<dim> const &                            dof_handler,
                                  LinearAlgebra::distributed::Vector<Number> const & solution,
                                  unsigned int const &        global_dof_index,
                                  std::vector<Number> const & fe_shape_values,
                                  Number &                    result)
{
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  Number const * sol_ptr = solution.begin() + global_dof_index;

  for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    result += sol_ptr[i] * fe_shape_values[i];
}


#endif /* INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_ */
