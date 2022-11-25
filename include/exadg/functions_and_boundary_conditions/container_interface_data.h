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

#ifndef INCLUDE_EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_CONTAINER_INTERFACE_DATA_H_
#define INCLUDE_EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_CONTAINER_INTERFACE_DATA_H_

// deal.II
#include <deal.II/base/tensor.h>

#include <map>
#include <tuple>
#include <vector>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/n_components_to_rank.h>

namespace ExaDG
{
/**
 * A data structure storing quadrature point information for each quadrature point, where the type
 * of data stored for each q-point is dealii::Tensor<rank, dim, number_type>.
 */
template<int rank, int dim, typename number_type>
class CouplingDataBase
{
public:
  typedef dealii::Tensor<rank, dim, number_type> data_type;

protected:
  static unsigned int const n_components = rank_to_n_components<rank, dim>();

  using quad_index = unsigned int;

  using ArrayQuadraturePoints = std::vector<dealii::Point<dim>>;

  using ArraySolutionValues = std::vector<data_type>;

public:
  std::vector<quad_index> const &
  get_quad_indices()
  {
    return quad_indices;
  }

  ArrayQuadraturePoints &
  get_array_q_points(quad_index const & q_index)
  {
    return map_q_points[q_index];
  }

  ArraySolutionValues &
  get_array_solution(quad_index const & q_index)
  {
    return map_solution[q_index];
  }

protected:
  std::vector<quad_index> quad_indices;

  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points;
  mutable std::map<quad_index, ArraySolutionValues>   map_solution;
};

/**
 * For this derived class, quadrature points are given by all the quadrature points on the boundary
 * faces of a mesh belonging to a given set of boundary IDs.
 */
template<int rank, int dim, typename number_type>
class CouplingDataSurface : public CouplingDataBase<rank, dim, number_type>
{
public:
  using data_type = typename CouplingDataBase<rank, dim, number_type>::data_type;

private:
  using quad_index = typename CouplingDataBase<rank, dim, number_type>::quad_index;

  using ArrayQuadraturePoints =
    typename CouplingDataBase<rank, dim, number_type>::ArrayQuadraturePoints;

  using ArraySolutionValues =
    typename CouplingDataBase<rank, dim, number_type>::ArraySolutionValues;

  using SetBoundaryIDs = std::set<dealii::types::boundary_id>;

  using Id = std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/>;

  using MapVectorIndex = std::map<Id, dealii::types::global_dof_index>;

public:
  CouplingDataSurface();

  /**
   * This function loops over the boundary faces of a triangulation using dealii::MatrixFree
   * infrastructure. For those faces belonging to a specified set of boundary IDs, all quadrature
   * points of that face are stored in global vector of quadrature points. Moreover, a map is
   * constructed that uniquely maps the face, q-point, and vectorized array index to a global index
   * used to access the global q-point and solution vectors. The solution vector is initialized
   * according with default values according to the global number of q-points.
   */
  template<typename Number>
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_,
        unsigned int const                               dof_index_,
        std::vector<quad_index> const &                  quad_indices_,
        SetBoundaryIDs const &                           set_bids_)
  {
    this->quad_indices = quad_indices_;

    for(auto q_index : this->quad_indices)
    {
      // initialize maps
      map_vector_index.emplace(q_index, MapVectorIndex());
      this->map_q_points.emplace(q_index, ArrayQuadraturePoints());
      this->map_solution.emplace(q_index, ArraySolutionValues());

      MapVectorIndex &        map_index          = map_vector_index.find(q_index)->second;
      ArrayQuadraturePoints & array_q_points_dst = this->map_q_points.find(q_index)->second;
      ArraySolutionValues &   array_solution_dst = this->map_solution.find(q_index)->second;

      // create map "ID = {face, q, v} <-> vector_index" and fill array of quadrature points
      for(unsigned int face = matrix_free_->n_inner_face_batches();
          face < matrix_free_->n_inner_face_batches() + matrix_free_->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(set_bids_.find(matrix_free_->get_boundary_id(face)) != set_bids_.end())
        {
          FaceIntegrator<dim, this->n_components, Number> integrator(*matrix_free_,
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

      array_solution_dst.resize(array_q_points_dst.size(), data_type());
    }
  }

  /**
   * This function returns the data item of the global vector of solution values given the
   * identification parameters that uniquely define a quadrature point.
   */
  data_type
  get_data(unsigned int const q_index,
           unsigned int const face,
           unsigned int const q,
           unsigned int const v) const;

private:
  mutable std::map<quad_index, MapVectorIndex> map_vector_index;
};

/**
 * For this derived class, quadrature points are given by all the quadrature points of a
 * triangulation. Depending on the type of discretization approach, the quadrature points might be
 * those inside cells/elements, but also those on the interior/boundary faces of a triangulation.
 */
template<int rank, int dim, typename number_type>
class CouplingDataVolume : public CouplingDataBase<rank, dim, number_type>
{
  // TODO: fill this class
};
} // namespace ExaDG



#endif /* INCLUDE_EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_CONTAINER_INTERFACE_DATA_H_ */
