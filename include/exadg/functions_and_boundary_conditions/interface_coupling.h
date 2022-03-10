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
#include <exadg/functions_and_boundary_conditions/function_cached.h>
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
class ContainerInterfaceData
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  using MapBoundaryCondition =
    std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>>;

  using quad_index = unsigned int;

  using Id = std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/>;

  using MapVectorIndex = std::map<Id, dealii::types::global_dof_index>;

  using ArrayQuadraturePoints = std::vector<dealii::Point<dim>>;

  typedef typename FunctionCached<rank, dim, double>::value_type value_type;

  using ArraySolutionValues = std::vector<value_type>;

public:
  ContainerInterfaceData();

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_,
        unsigned int const                               dof_index_,
        std::vector<quad_index> const &                  quad_indices_,
        MapBoundaryCondition const &                     map_bc_);

  std::vector<quad_index> const &
  get_quad_indices();

  ArrayQuadraturePoints &
  get_array_q_points(quad_index const & q_index);

  ArraySolutionValues &
  get_array_solution(quad_index const & q_index);

private:
  std::vector<quad_index> quad_indices;

  mutable std::map<quad_index, MapVectorIndex>        map_vector_index;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points;
  mutable std::map<quad_index, ArraySolutionValues>   map_solution;
};

template<int dim, int n_components, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  using MapBoundaryCondition =
    std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>>;

  using quad_index = unsigned int;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  InterfaceCoupling();

  /**
   * setup() function.
   *
   * TODO: The part dealing with marked vertices needs to be generalized. Currently,
   * use an empty map_bc_src_ to deactivate the feature marked_vertices.
   */
  void
  setup(std::shared_ptr<ContainerInterfaceData<dim, n_components, Number>> interface_data_dst_,
        MapBoundaryCondition const &                                       map_bc_src_,
        dealii::DoFHandler<dim> const &                                    dof_handler_src_,
        dealii::Mapping<dim> const &                                       mapping_src_,
        double const                                                       tolerance_);

  void
  update_data(VectorType const & dof_vector_src);

private:
  /*
   * dst-side
   */
  std::shared_ptr<ContainerInterfaceData<dim, n_components, Number>> interface_data_dst;

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
