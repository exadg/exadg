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

namespace ExaDG
{
using namespace dealii;

template<int dim, int n_components, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  using VectorType = LinearAlgebra::distributed::Vector<Number>;
#if DEAL_II_VERSION_GTE(10, 0, 0)
#else
  using VectorTypeDouble = LinearAlgebra::distributed::Vector<double>;
#endif
  using Integrator = FaceIntegrator<dim, n_components, Number>;

  using MapBoundaryCondition =
    std::map<types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>>;

  using quad_index = unsigned int;

  using Id = std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/>;

  using MapIndex = std::map<Id, types::global_dof_index>;

  using ArrayQuadraturePoints = std::vector<Point<dim>>;
  using ArrayTensor           = std::vector<Tensor<rank, dim, double>>;

public:
  InterfaceCoupling() : dof_handler_src(nullptr)
  {
  }

  void
  setup(std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst_in,
        unsigned int const                       dof_index_dst_in,
        std::vector<quad_index> const &          quad_indices_dst_in,
        MapBoundaryCondition const &             map_bc_in,
        DoFHandler<dim> const &                  dof_handler_src_in,
        Mapping<dim> const &                     mapping_src_in,
        VectorType const &                       dof_vector_src_in,
        double const                             tolerance_in)
  {
    quad_rules_dst  = quad_indices_dst_in;
    map_bc          = map_bc_in;
    dof_handler_src = &dof_handler_src_in;

#if DEAL_II_VERSION_GTE(10, 0, 0)
    // mark vertices at interface in order to make search of active cells around point more
    // efficient
    std::vector<bool> marked_vertices(dof_handler_src_in.get_triangulation().n_vertices(), false);
    for(auto const & cell : dof_handler_src_in.get_triangulation().active_cell_iterators())
    {
      if(!cell->is_artificial() && cell->at_boundary())
      {
        for(unsigned int const f : cell->face_indices())
        {
          if(cell->face(f)->at_boundary())
          {
            if(map_bc.find(cell->face(f)->boundary_id()) != map_bc.end())
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
#endif

    for(auto quadrature : quad_rules_dst)
    {
      // initialize maps
      map_index_dst.emplace(quadrature, MapIndex());
      map_q_points_dst.emplace(quadrature, ArrayQuadraturePoints());
      map_solution_dst.emplace(quadrature, ArrayTensor());

      MapIndex &              map_index          = map_index_dst.find(quadrature)->second;
      ArrayQuadraturePoints & array_q_points_dst = map_q_points_dst.find(quadrature)->second;
      ArrayTensor &           array_solution_dst = map_solution_dst.find(quadrature)->second;


      /*
       * 1. Setup: create map "ID = {face, q, v} <-> vector_index" and fill array of quadrature
       * points
       */
      for(unsigned int face = matrix_free_dst_in->n_inner_face_batches();
          face < matrix_free_dst_in->n_inner_face_batches() +
                   matrix_free_dst_in->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(map_bc.find(matrix_free_dst_in->get_boundary_id(face)) != map_bc.end())
        {
          Integrator integrator(*matrix_free_dst_in, true, dof_index_dst_in, quadrature);
          integrator.reinit(face);

          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
            {
              Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][v];

              Id                      id    = std::make_tuple(face, q, v);
              types::global_dof_index index = array_q_points_dst.size();
              map_index.emplace(id, index);
              array_q_points_dst.push_back(q_point);
            }
          }
        }
      }

      array_solution_dst.resize(array_q_points_dst.size(), Tensor<rank, dim, double>());

      /*
       * 2. Communication: exchange quadarature points with their owners
       */
      map_evaluator.emplace(quadrature,
                            Utilities::MPI::RemotePointEvaluation<dim>(tolerance_in,
                                                                       false
#if DEAL_II_VERSION_GTE(10, 0, 0)
                                                                       ,
                                                                       0,
                                                                       [marked_vertices]() {
                                                                         return marked_vertices;
                                                                       }
#endif
                                                                       ));

      map_evaluator[quadrature].reinit(array_q_points_dst,
                                       dof_handler_src_in.get_triangulation(),
                                       mapping_src_in);
    }

    // finally, give boundary condition access to the data
    for(auto boundary : map_bc)
    {
      boundary.second->set_data_pointer(map_index_dst, map_solution_dst);
    }

    update_data(dof_vector_src_in); // TODO: needed?
  }

  void
  update_data(VectorType const & dof_vector_src)
  {
#if DEAL_II_VERSION_GTE(10, 0, 0)
    dof_vector_src.update_ghost_values();
#else
    VectorTypeDouble dof_vector_src_double;
    dof_vector_src_double = dof_vector_src;
    dof_vector_src_double.update_ghost_values();
#endif

    for(auto quadrature : quad_rules_dst)
    {
#if DEAL_II_VERSION_GTE(10, 0, 0)
      std::vector<Tensor<rank, dim, Number>> const result =
        VectorTools::point_values<n_components>(map_evaluator[quadrature],
                                                *dof_handler_src,
                                                dof_vector_src,
                                                VectorTools::EvaluationFlags::avg);
#else
      std::vector<Tensor<rank, dim, double>> const result =
        VectorTools::point_values<n_components>(map_evaluator[quadrature],
                                                *dof_handler_src,
                                                dof_vector_src_double,
                                                VectorTools::EvaluationFlags::avg);
#endif

      for(unsigned int i = 0; i < result.size(); ++i)
        map_solution_dst[quadrature][i] = result[i];
    }
  }

private:
  /*
   * dst-side
   */
  std::vector<quad_index> quad_rules_dst;

  mutable std::map<quad_index, MapIndex>              map_index_dst;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points_dst;
  mutable std::map<quad_index, ArrayTensor>           map_solution_dst;

  std::map<quad_index, Utilities::MPI::RemotePointEvaluation<dim>> map_evaluator;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>> map_bc;

  /*
   * src-side
   */
  DoFHandler<dim> const * dof_handler_src;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
