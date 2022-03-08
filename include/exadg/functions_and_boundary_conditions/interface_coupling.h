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
  using ArraySolutionValues   = std::vector<dealii::Tensor<rank, dim, double>>;

public:
  ContainerInterfaceData()
  {
  }

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_,
        unsigned int const                               dof_index_,
        std::vector<quad_index> const &                  quad_indices_,
        MapBoundaryCondition const &                     map_bc_)
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
        if(map_bc_.find(matrix_free_->get_boundary_id(face)) != map_bc_.end())
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

      array_solution_dst.resize(array_q_points_dst.size(), dealii::Tensor<rank, dim, double>());
    }

    // finally, give boundary condition access to the data
    for(auto boundary : map_bc_)
    {
      boundary.second->set_data_pointer(map_vector_index, map_solution);
    }
  }

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
  InterfaceCoupling() : dof_handler_src(nullptr)
  {
  }

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
        double const                                                       tolerance_)
  {
    interface_data_dst = interface_data_dst_;
    dof_handler_src    = &dof_handler_src_;

#if DEAL_II_VERSION_GTE(10, 0, 0)
    // mark vertices at interface in order to make search of active cells around point more
    // efficient
    std::vector<bool> marked_vertices(dof_handler_src_.get_triangulation().n_vertices(), false);

    for(auto const & cell : dof_handler_src_.get_triangulation().active_cell_iterators())
    {
      if(!cell->is_artificial() && cell->at_boundary())
      {
        for(unsigned int const f : cell->face_indices())
        {
          if(cell->face(f)->at_boundary())
          {
            if(map_bc_src_.find(cell->face(f)->boundary_id()) != map_bc_src_.end())
            {
              for(unsigned int const v : cell->face(f)->vertex_indices())
              {
                marked_vertices[cell->face(f)->vertex_index(v)] = true;
              }
            }
          }
        }
      }
    }

    // To improve robustness, make sure that not all entries of marked_vertices are false.
    // Otherwise, points will simply not be found by RemotePointEvaluation and results will
    // probably be wrong.
    if (std::all_of(marked_vertices.begin(), marked_vertices.end(), [](bool marked){return marked == false;}))
      std::fill(marked_vertices.begin(), marked_vertices.end(), true);
#endif

    for(auto quad_index : interface_data_dst->get_quad_indices())
    {
      // exchange quadrature points with their owners
      map_evaluator.emplace(quad_index,
                            dealii::Utilities::MPI::RemotePointEvaluation<dim>(
                              tolerance_,
                              false
#if DEAL_II_VERSION_GTE(10, 0, 0)
                              ,
                              0,
                              [marked_vertices]() { return marked_vertices; }
#endif
                              ));

      map_evaluator[quad_index].reinit(interface_data_dst->get_array_q_points(quad_index),
                                       dof_handler_src_.get_triangulation(),
                                       mapping_src_);
    }
  }

  void
  update_data(VectorType const & dof_vector_src)
  {
#if DEAL_II_VERSION_GTE(10, 0, 0)
    dof_vector_src.update_ghost_values();
#else
    dealii::LinearAlgebra::distributed::Vector<double> dof_vector_src_double;
    dof_vector_src_double = dof_vector_src;
    dof_vector_src_double.update_ghost_values();
#endif

    for(auto quadrature : interface_data_dst->get_quad_indices())
    {
#if DEAL_II_VERSION_GTE(10, 0, 0)
      std::vector<dealii::Tensor<rank, dim, Number>> const result =
        dealii::VectorTools::point_values<n_components>(map_evaluator[quadrature],
                                                        *dof_handler_src,
                                                        dof_vector_src,
                                                        dealii::VectorTools::EvaluationFlags::avg);
#else
      std::vector<dealii::Tensor<rank, dim, double>> const result =
        dealii::VectorTools::point_values<n_components>(map_evaluator[quadrature],
                                                        *dof_handler_src,
                                                        dof_vector_src_double,
                                                        dealii::VectorTools::EvaluationFlags::avg);
#endif

      auto & array_solution = interface_data_dst->get_array_solution(quadrature);
      for(unsigned int i = 0; i < result.size(); ++i)
        array_solution[i] = result[i];
    }
  }

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
