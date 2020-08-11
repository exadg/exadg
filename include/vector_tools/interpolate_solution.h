/*
 * interpolate_solution.h
 *
 *  Created on: Jul 28, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_
#define INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/la_parallel_vector.h>

/*
 * For a given vector of adjacent cells and points in reference coordinates, determine
 * and return the dof_indices and shape_values to be used later for interpolation of the
 * solution.
 */
template<int dim, typename Number>
std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>
get_dof_indices_and_shape_values(
  std::vector<std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>> const &
                                                     adjacent_cells,
  DoFHandler<dim> const &                            dof_handler,
  Mapping<dim> const &                               mapping,
  LinearAlgebra::distributed::Vector<Number> const & solution)
{
  // fill dof_indices_and_shape_values
  std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>
    dof_indices_and_shape_values;

  for(auto cell_tria : adjacent_cells)
  {
    // transform Triangulation::active_cell_iterator into DoFHandler::active_cell_iterator,
    // see constructor of DoFCellAccessor
    typename DoFHandler<dim>::active_cell_iterator cell_dof = {&dof_handler.get_triangulation(),
                                                               cell_tria.first->level(),
                                                               cell_tria.first->index(),
                                                               &dof_handler};

    typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> PairDof;
    PairDof cell_and_point(cell_dof, cell_tria.second);

    // go on only if cell is owned by the processor
    if(cell_and_point.first->is_locally_owned())
    {
      const Quadrature<dim> quadrature(
        GeometryInfo<dim>::project_to_unit_cell(cell_and_point.second));

      const FiniteElement<dim> & fe = dof_handler.get_fe();
      FEValues<dim>              fe_values(mapping, fe, quadrature, update_values);
      fe_values.reinit(cell_and_point.first);
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell_and_point.first->get_dof_indices(dof_indices);

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
 * Interpolate solution from dof vector by using precomputed shape function values.
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


#endif /* INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_ */
