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

#ifndef INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_
#define INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/*
 * For a given vector of adjacent cells and points in reference coordinates, determine
 * and return the dof_indices and shape_values to be used later for interpolation of the
 * solution.
 */
template<int dim, typename Number>
std::vector<std::pair<std::vector<dealii::types::global_dof_index>, std::vector<Number>>>
get_dof_indices_and_shape_values(
  std::vector<std::pair<typename dealii::Triangulation<dim>::active_cell_iterator,
                        dealii::Point<dim>>> const &         adjacent_cells,
  dealii::DoFHandler<dim> const &                            dof_handler,
  dealii::Mapping<dim> const &                               mapping,
  dealii::LinearAlgebra::distributed::Vector<Number> const & solution)
{
  // fill dof_indices_and_shape_values
  std::vector<std::pair<std::vector<dealii::types::global_dof_index>, std::vector<Number>>>
    dof_indices_and_shape_values;

  for(auto cell_tria : adjacent_cells)
  {
    // transform Triangulation::active_cell_iterator into dealii::DoFHandler::active_cell_iterator,
    // see constructor of DoFCellAccessor
    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dof = {
      &dof_handler.get_triangulation(),
      cell_tria.first->level(),
      cell_tria.first->index(),
      &dof_handler};

    typedef std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>
            PairDof;
    PairDof cell_and_point(cell_dof, cell_tria.second);

    // go on only if cell is owned by the processor
    if(cell_and_point.first->is_locally_owned())
    {
      const dealii::Quadrature<dim> quadrature(
        dealii::GeometryInfo<dim>::project_to_unit_cell(cell_and_point.second));

      const dealii::FiniteElement<dim> & fe = dof_handler.get_fe();
      dealii::FEValues<dim>              fe_values(mapping, fe, quadrature, dealii::update_values);
      fe_values.reinit(cell_and_point.first);
      std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell_and_point.first->get_dof_indices(dof_indices);

      std::vector<dealii::types::global_dof_index> dof_indices_global(fe.dofs_per_cell);
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
    dealii::Tensor<rank, dim, Number>
    value(dealii::DoFHandler<dim> const &                            dof_handler,
          dealii::LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<dealii::types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                                fe_shape_values)
  {
    (void)dof_handler;
    (void)solution;
    (void)dof_indices;
    (void)fe_shape_values;

    AssertThrow(false, dealii::ExcMessage("not implemented."));

    dealii::Tensor<rank, dim, Number> result;
    return result;
  }
};

/*
 * The quantity to be evaluated is of type dealii::Tensor<0,dim,Number>.
 */
template<int dim, typename Number>
struct Interpolator<0, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, Number>
    value(dealii::DoFHandler<dim> const &                            dof_handler,
          dealii::LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<dealii::types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                                fe_shape_values)
  {
    Assert(dof_handler.get_fe().dofs_per_cell == fe_shape_values.size(),
           dealii::ExcMessage("Vector fe_shape_values has wrong size."));

    Number result = Number(0.0);

    dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      result += solution.local_element(dof_indices[i]) * fe_shape_values[i];

    dealii::Tensor<0, dim, Number> result_tensor = result;

    return result_tensor;
  }
};

/*
 * The quantity to be evaluated is of type dealii::Tensor<1, dim, Number>.
 */
template<int dim, typename Number>
struct Interpolator<1, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, Number>
    value(dealii::DoFHandler<dim> const &                            dof_handler,
          dealii::LinearAlgebra::distributed::Vector<Number> const & solution,
          std::vector<dealii::types::global_dof_index> const &       dof_indices,
          std::vector<Number> const &                                fe_shape_values)
  {
    Assert(dof_handler.get_fe().dofs_per_cell == fe_shape_values.size(),
           dealii::ExcMessage("Vector fe_shape_values has wrong size."));

    dealii::Tensor<1, dim, Number> result;

    dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      result[fe.system_to_component_index(i).first] +=
        solution.local_element(dof_indices[i]) * fe_shape_values[i];

    return result;
  }
};

} // namespace ExaDG

#endif /* INCLUDE_VECTOR_TOOLS_INTERPOLATE_SOLUTION_H_ */
