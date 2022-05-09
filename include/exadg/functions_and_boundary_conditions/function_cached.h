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

#ifndef INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_
#define INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_

// deal.II
#include <deal.II/base/tensor.h>

#include <map>
#include <tuple>
#include <vector>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
template<int rank, int dim>
class ContainerInterfaceData
{
public:
  // this is the underlying data type for FunctionCached.
  typedef dealii::Tensor<rank, dim, double> value_type;

private:
  static unsigned int const n_components =
    (rank == 0) ? 1 : ((rank == 1) ? dim : dealii::numbers::invalid_unsigned_int);

  using SetBoundaryIDs = std::set<dealii::types::boundary_id>;

  using quad_index = unsigned int;

  using Id = std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/>;

  using MapVectorIndex = std::map<Id, dealii::types::global_dof_index>;

  using ArrayQuadraturePoints = std::vector<dealii::Point<dim>>;

  using ArraySolutionValues = std::vector<value_type>;

public:
  ContainerInterfaceData();

  template<typename Number>
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_,
        unsigned int const                               dof_index_,
        std::vector<quad_index> const &                  quad_indices_,
        SetBoundaryIDs const &                           set_bids_)
  {
    quad_indices = quad_indices_;

    for(auto q_index : quad_indices)
    {
      // initialize maps
      map_vector_index.emplace(q_index, MapVectorIndex());
      map_q_points.emplace(q_index, ArrayQuadraturePoints());
      map_solution.emplace(q_index, ArraySolutionValues());

      MapVectorIndex &        map_index          = map_vector_index.find(q_index)->second;
      ArrayQuadraturePoints & array_q_points_dst = map_q_points.find(q_index)->second;
      ArraySolutionValues &   array_solution_dst = map_solution.find(q_index)->second;

      // create map "ID = {face, q, v} <-> vector_index" and fill array of quadrature points
      for(unsigned int face = matrix_free_->n_inner_face_batches();
          face < matrix_free_->n_inner_face_batches() + matrix_free_->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(set_bids_.find(matrix_free_->get_boundary_id(face)) != set_bids_.end())
        {
          FaceIntegrator<dim, n_components, Number> integrator(*matrix_free_,
                                                               true,
                                                               dof_index_,
                                                               q_index);
          integrator.reinit(face);

          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            dealii::Point<dim, dealii::VectorizedArray<Number>> q_points =
              integrator.quadrature_point(q);

            for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
            {
              dealii::Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][v];

              Id                              id    = std::make_tuple(face, q, v);
              dealii::types::global_dof_index index = array_q_points_dst.size();
              map_index.emplace(id, index);
              array_q_points_dst.push_back(q_point);
            }
          }
        }
      }

      array_solution_dst.resize(array_q_points_dst.size(), value_type());
    }
  }

  std::vector<quad_index> const &
  get_quad_indices();

  ArrayQuadraturePoints &
  get_array_q_points(quad_index const & q_index);

  ArraySolutionValues &
  get_array_solution(quad_index const & q_index);

  value_type
  get_data(unsigned int const q_index,
           unsigned int const face,
           unsigned int const q,
           unsigned int const v) const;

private:
  std::vector<quad_index> quad_indices;

  mutable std::map<quad_index, MapVectorIndex>        map_vector_index;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points;
  mutable std::map<quad_index, ArraySolutionValues>   map_solution;
};

/*
 * The only reason why we do not integrate ContainerInterfaceData directly into
 * FunctionCached is that we want to use only one object of type
 * ContainerInterfaceData for (potentially) many boundary IDs and, therefore,
 * many objects of type FunctionCached.
 */
template<int rank, int dim>
class FunctionCached
{
private:
  typedef typename ContainerInterfaceData<rank, dim>::value_type value_type;

public:
  FunctionCached();

  // read data
  value_type
  tensor_value(unsigned int const face,
               unsigned int const q,
               unsigned int const v,
               unsigned int const quad_index) const;

  // initialize data pointer
  void
  set_data_pointer(std::shared_ptr<ContainerInterfaceData<rank, dim>> const interface_data_);

private:
  std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
