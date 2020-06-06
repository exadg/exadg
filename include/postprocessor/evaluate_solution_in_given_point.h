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

template<int dim>
unsigned int
n_locally_owned_active_cells_around_point(const Triangulation<dim> & tria,
                                          const Mapping<dim> &       mapping,
                                          const Point<dim> &         point,
                                          const double               tolerance)
{
  using Pair = std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>;

  std::vector<Pair> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, tria, point, tolerance);

  // count locally owned active cells
  unsigned int counter = 0;
  for(auto cell : adjacent_cells)
  {
    if(cell.first->is_locally_owned())
    {
      ++counter;
    }
  }

  return counter;
}

/*
 * For a given point in physical space, find all locally owned, adjacent cells and store the
 * dof index of the first dof of the cell as well as the shape function values of all dofs
 * (which can then be used for interpolation of the solution in the given point afterwards).
 * The tuple (dof_index, shape_values) is stored in a vector where each entry corresponds
 * to one adjacent, locally-owned cell.
 */
template<int dim, typename Number>
std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>
get_dof_indices_and_shape_values(DoFHandler<dim> const &                            dof_handler,
                                 Mapping<dim> const &                               mapping,
                                 LinearAlgebra::distributed::Vector<Number> const & solution,
                                 Point<dim> const &                                 point,
                                 double const tolerance = 1.e-10)
{
  std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>
    dof_indices_and_shape_values;

  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> Pair;

  std::vector<Pair> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping, dof_handler, point, tolerance);

  // loop over all adjacent cells
  for(auto cell : adjacent_cells)
  {
    // go on only if cell is owned by the processor
    if(cell.first->is_locally_owned())
    {
      const Quadrature<dim> quadrature(GeometryInfo<dim>::project_to_unit_cell(cell.second));

      const FiniteElement<dim> & fe = dof_handler.get_fe();
      FEValues<dim>              fe_values(mapping, fe, quadrature, update_values);
      fe_values.reinit(cell.first);
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell.first->get_dof_indices(dof_indices);

      std::vector<types::global_dof_index> dof_indices_global(fe.dofs_per_cell);
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        dof_indices_global[i] = solution.get_partitioner()->global_to_local(dof_indices[i]);

      std::vector<Number> fe_shape_values(fe.dofs_per_cell);
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        fe_shape_values[i] = fe_values.shape_value(i, 0);

      dof_indices_and_shape_values.emplace_back(dof_indices_global, fe_shape_values);
    }
  }

  return dof_indices_and_shape_values;
}

/*
 * Interpolate solution from dof vector by using precomputed shape functions values (for
 * efficiency!). Note that we assume that we are dealing with discontinuous finite elements.
 */
template<int rank, int dim, typename Number>
struct Interpolator
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, Number>
    value(DoFHandler<dim> const &                            dof_handler,
          LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                        fe_shape_values)
  {
    (void)dof_handler;
    (void)solution;
    (void)dof_indices;
    (void)fe_shape_values;

    AssertThrow(false, ExcMessage("not implemented."));

    Tensor<rank, dim, Number> result;
    return result;
  }
};

/*
 * The quantity to be evaluated is of type Tensor<0,dim,Number>.
 */
template<int dim, typename Number>
struct Interpolator<0, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<0, dim, Number>
    value(DoFHandler<dim> const &                            dof_handler,
          LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                        fe_shape_values)
  {
    Assert(dof_handler.get_fe().dofs_per_cell == fe_shape_values.size(),
           ExcMessage("Vector fe_shape_values has wrong size."));

    Number result = Number(0.0);

    FiniteElement<dim> const & fe = dof_handler.get_fe();
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      result += solution.local_element(dof_indices[i]) * fe_shape_values[i];

    Tensor<0, dim, Number> result_tensor = result;

    return result_tensor;
  }
};

/*
 * The quantity to be evaluated is of type Tensor<1, dim, Number>.
 */
template<int dim, typename Number>
struct Interpolator<1, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    value(DoFHandler<dim> const &                            dof_handler,
          LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                        fe_shape_values)
  {
    Assert(dof_handler.get_fe().dofs_per_cell == fe_shape_values.size(),
           ExcMessage("Vector fe_shape_values has wrong size."));

    Tensor<1, dim, Number> result;

    FiniteElement<dim> const & fe = dof_handler.get_fe();
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      result[fe.system_to_component_index(i).first] +=
        solution.local_element(dof_indices[i]) * fe_shape_values[i];

    return result;
  }
};


#endif /* INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_ */
