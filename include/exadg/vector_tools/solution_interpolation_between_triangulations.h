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

#ifndef INCLUDE_VECTOR_TOOLS_SOLUTION_INTERPOLATION_BETWEEN_TRIANGULATIONS_H
#define INCLUDE_VECTOR_TOOLS_SOLUTION_INTERPOLATION_BETWEEN_TRIANGULATIONS_H

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/vector_tools.h>

namespace ExaDG
{
/**
 * Class to transfer solutions between different DoFHandlers via interpolation. This class requires,
 * that the destination DofHandler has generalized support points.
 */
template<int dim>
class SolutionInterpolationBetweenTriangulations
{
public:
  /**
   * Set source and target triangulations on which solution transfer is performed.
   *
   * @param[in] dof_handler_dst_in Target DofHandler.
   * @param[in] mapping_dst_in Target Mapping.
   * @param[in] dof_handler_src_in Source DofHandler.
   * @param[in] mapping_src_in Source Mapping.
   */
  void
  reinit(dealii::DoFHandler<dim> const & dof_handler_dst_in,
         dealii::Mapping<dim> const &    mapping_dst_in,
         dealii::DoFHandler<dim> const & dof_handler_src_in,
         dealii::Mapping<dim> const &    mapping_src_in)
  {
    AssertThrow(
      dof_handler_dst_in.get_fe().has_generalized_support_points(),
      dealii::ExcMessage(
        "Solution can only be interpolated to finite elements that have support points."));

    dof_handler_dst = &dof_handler_dst_in;
    dof_handler_src = &dof_handler_src_in;

    rpe.reinit(collect_mapped_support_points(dof_handler_dst_in, mapping_dst_in),
               dof_handler_src_in.get_triangulation(),
               mapping_src_in);
  }

  /**
   * Interpolate the solution from a source to a target triangulation, that were set with reinit.
   * At support points which are not overlapping the source triangulation, corresponding dof entries
   * are set to 0.
   *
   * @param[in] dst Target Dof Vector.
   * @param[in] src Source Dof Vector.
   */
  template<int n_components, typename VectorType1, typename VectorType2>
  void
  interpolate_solution(VectorType1 &                                               dst,
                       VectorType2 const &                                         src,
                       dealii::VectorTools::EvaluationFlags::EvaluationFlags const flags =
                         dealii::VectorTools::EvaluationFlags::avg) const
  {
    AssertThrow(src.size() == dof_handler_src->n_dofs(),
                dealii::ExcMessage("Dimensions do not fit."));
    AssertThrow(dst.size() == dof_handler_dst->n_dofs(),
                dealii::ExcMessage("Dimensions do not fit."));

    src.update_ghost_values();
    auto const values =
      dealii::VectorTools::point_values<n_components>(rpe, *dof_handler_src, src, flags);
    src.zero_out_ghost_values();

    fill_dof_vector_with_values<n_components>(dst, *dof_handler_dst, values);
  }

private:
  std::vector<dealii::Point<dim>>
  collect_mapped_support_points(dealii::DoFHandler<dim> const & dof_handler,
                                dealii::Mapping<dim> const &    mapping) const
  {
    std::vector<dealii::Point<dim>> support_points;

    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(cell->is_locally_owned() == false)
        continue;

      auto cellwise_support_points = cell->get_fe().get_generalized_support_points();

      for(auto & csp : cellwise_support_points)
        csp = mapping.transform_unit_to_real_cell(cell, csp);

      support_points.insert(support_points.end(),
                            cellwise_support_points.begin(),
                            cellwise_support_points.end());
    }

    return support_points;
  }

  template<int n_components, typename VectorType, typename value_type>
  void
  fill_dof_vector_with_values(VectorType &                    dst,
                              dealii::DoFHandler<dim> const & dof_handler,
                              std::vector<value_type> const & values) const
  {
    auto ptr = values.begin();
    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(cell->is_locally_owned() == false)
        continue;

      auto const & fe = cell->get_fe();

      std::vector<double> dof_values(fe.n_dofs_per_cell());
      unsigned int const  n_support_points = fe.get_generalized_support_points().size();
      std::vector<dealii::Vector<double>> component_dof_values(
        n_support_points, dealii::Vector<double>(n_components));

      for(unsigned int i = 0; i < n_support_points; ++i)
      {
        if constexpr(n_components == 1)
          component_dof_values[i] = *ptr;
        else
          component_dof_values[i] =
            std::move(dealii::Vector<double>(ptr->begin_raw(), ptr->end_raw()));

        ++ptr;
      }

      fe.convert_generalized_support_point_values_to_dof_values(component_dof_values, dof_values);

      cell->set_dof_values(dealii::Vector<typename VectorType::value_type>(dof_values.begin(),
                                                                           dof_values.end()),
                           dst);
    }
  }

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_dst;
  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_src;
  dealii::Utilities::MPI::RemotePointEvaluation<dim>  rpe;
};

} // namespace ExaDG

#endif /* INCLUDE_VECTOR_TOOLS_SOLUTION_INTERPOLATION_BETWEEN_TRIANGULATIONS_H */
