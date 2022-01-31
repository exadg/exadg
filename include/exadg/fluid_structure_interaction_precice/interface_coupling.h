/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_INTERFACE_COUPLING_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_INTERFACE_COUPLING_H_

// deal.II
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>
#include <exadg/grid/find_all_active_cells_around_point.h>
#include <exadg/vector_tools/interpolate_solution.h>

namespace ExaDG
{
using namespace dealii;


template<int dim, int n_components, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef InterfaceCoupling<dim, n_components, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;
  typedef FaceIntegrator<dim, n_components, Number>  Integrator;
  typedef std::pair<unsigned int, unsigned int>      Range;

  typedef std::map<types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>>
    MapBoundaryCondition;

  typedef unsigned int quad_index;
  typedef unsigned int mpi_rank;

  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;

  typedef std::map<Id, types::global_dof_index> MapIndex;

  typedef std::pair<std::vector<types::global_dof_index>, std::vector<double>> Cache;

  typedef std::vector<Point<dim>> ArrayQuadraturePoints;
  using ArrayTensor = std::vector<dealii::Tensor<rank, dim, double>>;

public:
  InterfaceCoupling(std::shared_ptr<Adapter::Adapter<dim, dim, VectorType>> precice)
    : precice(precice)
  {
  }

  void
  setup(std::shared_ptr<const MatrixFree<dim, Number>> matrix_free_dst,
        unsigned int const                             dof_index_dst,
        std::vector<quad_index> const &                quad_indices_dst_in,
        MapBoundaryCondition const &                   map_bc_in,
        const std::string &                            mesh_name,
        const std::string &                            data_name)
  {
    quad_rules_dst = quad_indices_dst_in;
    map_bc         = map_bc_in;

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
      for(unsigned int face = matrix_free_dst->n_inner_face_batches();
          face <
          matrix_free_dst->n_inner_face_batches() + matrix_free_dst->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(map_bc.find(matrix_free_dst->get_boundary_id(face)) != map_bc.end())
        {
          Integrator integrator(*matrix_free_dst, true, dof_index_dst, quadrature);
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
      array_solution_dst.resize(array_q_points_dst.size());
    }

    ArrayQuadraturePoints map_q_points_dst_precice;

    for(auto quadrature : quad_rules_dst)
    {
      ArrayQuadraturePoints & array_q_points_dst = map_q_points_dst.find(quadrature)->second;

      for(const auto i : array_q_points_dst)
        map_q_points_dst_precice.emplace_back(i);
    }

    // finally, give boundary condition access to the data
    for(auto boundary : map_bc)
      boundary.second->set_data_pointer(map_index_dst, map_solution_dst);

    precice_mesh_name = mesh_name;
    precice->add_read_interface(map_q_points_dst_precice, matrix_free_dst, mesh_name, data_name);
  }

  void
  read()
  {
    // communicate with preCICE
    std::vector<Tensor<rank, dim, double>> array_solution_dst_precice =
      precice->read_block_data(precice_mesh_name);
    Assert(array_solution_dst_precice.size() > 0, ExcInternalError());
    Assert(precice_mesh_name != "", ExcInternalError());
    // extract values of each quadrature rule
    unsigned int c = 0;
    for(auto quadrature : quad_rules_dst)
    {
      ArrayTensor & array_solution_dst = map_solution_dst.find(quadrature)->second;

      for(unsigned int i = 0; i < array_solution_dst.size(); ++i, ++c)
        array_solution_dst[i] = array_solution_dst_precice[c];
    }
  }

private:
  std::shared_ptr<Adapter::Adapter<dim, dim, VectorType>> precice;

  std::vector<quad_index>                             quad_rules_dst;
  mutable std::map<quad_index, MapIndex>              map_index_dst;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points_dst;
  mutable std::map<quad_index, ArrayTensor>           map_solution_dst;
  mutable std::map<types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>> map_bc;
  std::string precice_mesh_name{};
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
