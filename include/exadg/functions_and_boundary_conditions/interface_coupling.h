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

#ifndef INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_
#define INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>
#include <exadg/utilities/n_components_to_rank.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
class SolutionInterpolator
{
private:
  static unsigned int const n_components = rank_to_n_components<rank, dim>();

  using quad_index = unsigned int;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  SolutionInterpolator();

  /**
   * setup() function.
   *
   * The aim of @param marked_vertices_src_ is to make the search of points on the src side
   * computationally more efficient. If no useful information can be provided for this parameter, an
   * empty vector has to be passed to this function.
   *
   * @param tolerance_ is a geometric tolerance passed to dealii::RemotePointEvaluation and used for
   * the search of points on the src side.
   */
  void
  setup(std::shared_ptr<CouplingDataBase<rank, dim, double>> coupling_data_dst_,
        dealii::DoFHandler<dim> const &                      dof_handler_src_,
        dealii::Mapping<dim> const &                         mapping_src_,
        std::vector<bool> const &                            marked_vertices_src_,
        double const                                         tolerance_);

  void
  update_data(VectorType const & dof_vector_src);

private:
  /*
   * dst-side
   */
  std::shared_ptr<CouplingDataBase<rank, dim, double>> coupling_data_dst;

  /*
   *  Evaluates solution on src-side in those points specified by dst-side
   */
  std::map<quad_index, dealii::Utilities::MPI::RemotePointEvaluation<dim>> map_evaluator;

  /*
   * src-side
   */
  dealii::DoFHandler<dim> const * dof_handler_src;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
