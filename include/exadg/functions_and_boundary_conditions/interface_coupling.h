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
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/numerics/rtree.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>
#include <exadg/grid/find_all_active_cells_around_point.h>
#include <exadg/vector_tools/interpolate_solution.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, int spacedim>
class InterfaceCommunicator
{
public:
  /*
   * Constructor.
   */
  InterfaceCommunicator(const std::vector<Point<spacedim>> &              quadrature_points,
                        const Triangulation<dim, spacedim> &              tria,
                        const Mapping<dim, spacedim> &                    mapping,
                        double const                                      tolerance,
                        const std::vector<bool> &                         marked_vertices,
                        std::shared_ptr<GridTools::Cache<dim, dim>> const cache)
    : comm(tria.get_communicator())
  {
    // create bounding boxed of local active cells
    std::vector<BoundingBox<spacedim>> local_boxes;
    for(auto const cell : tria.active_cell_iterators())
      if(cell->is_locally_owned())
        local_boxes.push_back(mapping.get_bounding_box(cell));

    // it might happen that there are no cells on a processor
    if(local_boxes.size() == 0)
      local_boxes.resize(1);

    // create r-tree of bounding boxes
    auto const local_tree = pack_rtree(local_boxes);

    // compress r-tree to a minimal set of bounding boxes
    auto const local_reduced_box = extract_rtree_level(local_tree, 0);

    // gather bounding boxes of other processes
    auto const global_bounding_boxes = Utilities::MPI::all_gather(comm, local_reduced_box);

    // determine ranks which might poses quadrature point
    auto points_per_process =
      std::vector<std::vector<Point<spacedim>>>(global_bounding_boxes.size());

    auto points_per_process_offset =
      std::vector<std::vector<unsigned int>>(global_bounding_boxes.size());

    for(unsigned int i = 0; i < quadrature_points.size(); ++i)
    {
      auto const & point = quadrature_points[i];
      for(unsigned rank = 0; rank < global_bounding_boxes.size(); ++rank)
        for(auto const & box : global_bounding_boxes[rank])
          if(box.point_inside(point, tolerance))
          {
            points_per_process[rank].emplace_back(point);
            points_per_process_offset[rank].emplace_back(i);
            break;
          }
    }

    // only communicate with processes that might have a quadrature point
    std::vector<unsigned int> targets;

    for(unsigned int i = 0; i < points_per_process.size(); ++i)
      if(points_per_process[i].size() > 0 && i != Utilities::MPI::this_mpi_process(comm))
        targets.emplace_back(i);


    std::map<unsigned int, std::vector<Point<spacedim>>> relevant_points_per_process;
    std::map<unsigned int, std::vector<unsigned int>>    relevant_points_per_process_offset;
    std::map<unsigned int, std::vector<unsigned int>>    relevant_points_per_process_count;


    // for local quadrature points no communication is needed...
    {
      unsigned int const my_rank = Utilities::MPI::this_mpi_process(comm);

      auto const & potentially_local_points = points_per_process[my_rank];


      {
        std::vector<Point<spacedim>> points;
        std::vector<unsigned int>    points_offset;
        std::vector<unsigned int>    count;

        auto const & potentially_relevant_points        = points_per_process[my_rank];
        auto const & potentially_relevant_points_offset = points_per_process_offset[my_rank];

        typename Triangulation<dim, dim>::active_cell_iterator cell_hint =
          typename Triangulation<dim, dim>::active_cell_iterator();
        for(unsigned int j = 0; j < potentially_local_points.size(); ++j)
        {
          unsigned int const counter =
            n_locally_owned_active_cells_around_point(tria,
                                                      mapping,
                                                      potentially_relevant_points[j],
                                                      tolerance,
                                                      cell_hint,
                                                      marked_vertices,
                                                      *cache);

          if(counter > 0)
          {
            points.push_back(potentially_relevant_points[j]);
            points_offset.push_back(potentially_relevant_points_offset[j]);
            count.push_back(counter);
          }
        }


        if(points.size() > 0)
        {
          relevant_remote_points_per_process[my_rank]       = points;
          relevant_remote_points_count_per_process[my_rank] = count;

          relevant_points_per_process[my_rank]        = points;
          relevant_points_per_process_offset[my_rank] = points_offset;
          relevant_points_per_process_count[my_rank]  = count;
          map_recv[my_rank]                           = points;
        }
      }
    }

    // send to remote ranks the requested quadrature points and eliminate not needed ones
    // (note: currently, we cannot communicate points -> switch to doubles here)
    Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<double, unsigned int> process(
      [&]() { return targets; },
      [&](unsigned int const other_rank, std::vector<double> & send_buffer) {
        // send requested points
        for(auto point : points_per_process[other_rank])
          for(unsigned int i = 0; i < spacedim; ++i)
            send_buffer.emplace_back(point[i]);
      },
      [&](unsigned int const &        other_rank,
          const std::vector<double> & recv_buffer,
          std::vector<unsigned int> & request_buffer) {
        // received points, determine if point is actually possessed, and
        // send the result back

        std::vector<Point<spacedim>> relevant_remote_points;
        std::vector<unsigned int>    relevant_remote_points_count;

        request_buffer.clear();
        request_buffer.resize(recv_buffer.size() / spacedim);

        typename Triangulation<dim, dim>::active_cell_iterator cell_hint =
          typename Triangulation<dim, dim>::active_cell_iterator();
        for(unsigned int i = 0, j = 0; i < recv_buffer.size(); i += spacedim, ++j)
        {
          Point<spacedim> point;
          for(unsigned int j = 0; j < spacedim; ++j)
            point[j] = recv_buffer[i + j];

          unsigned int const counter = n_locally_owned_active_cells_around_point(
            tria, mapping, point, tolerance, cell_hint, marked_vertices, *cache);

          request_buffer[j] = counter;

          if(counter > 0)
          {
            relevant_remote_points.push_back(point);
            relevant_remote_points_count.push_back(counter);
          }
        }

        if(relevant_remote_points.size() > 0)
        {
          relevant_remote_points_per_process[other_rank]       = relevant_remote_points;
          relevant_remote_points_count_per_process[other_rank] = relevant_remote_points_count;
        }
      },
      [&](unsigned int const other_rank, std::vector<unsigned int> & recv_buffer) {
        // prepare buffer
        recv_buffer.resize(points_per_process[other_rank].size());
      },
      [&](unsigned int const other_rank, const std::vector<unsigned int> & recv_buffer) {
        // store recv_buffer -> make the algorithm deterministic

        auto const & potentially_relevant_points        = points_per_process[other_rank];
        auto const & potentially_relevant_points_offset = points_per_process_offset[other_rank];

        std::vector<Point<spacedim>> points;
        std::vector<unsigned int>    points_offset;
        std::vector<unsigned int>    count;

        AssertDimension(potentially_relevant_points.size(), recv_buffer.size());
        AssertDimension(potentially_relevant_points_offset.size(), recv_buffer.size());

        for(unsigned int i = 0; i < recv_buffer.size(); ++i)
          if(recv_buffer[i] > 0)
          {
            points.push_back(potentially_relevant_points[i]);
            points_offset.push_back(potentially_relevant_points_offset[i]);
            count.push_back(recv_buffer[i]);
          }

        if(points.size() > 0)
        {
          relevant_points_per_process[other_rank]        = points;
          relevant_points_per_process_offset[other_rank] = points_offset;
          relevant_points_per_process_count[other_rank]  = count;
          map_recv[other_rank]                           = points;
        }
      });

    Utilities::MPI::ConsensusAlgorithms::Selector<double, unsigned int>(process, comm).run();

    quadrature_points_count.resize(quadrature_points.size(), 0);

    for(auto const & i : relevant_points_per_process)
    {
      unsigned int const rank = i.first;

      std::vector<std::pair<unsigned int, unsigned int>> indices;

      auto const & relevant_points        = relevant_points_per_process[rank];
      auto const & relevant_points_offset = relevant_points_per_process_offset[rank];
      auto const & relevant_points_count  = relevant_points_per_process_count[rank];

      for(unsigned int j = 0; j < relevant_points.size(); ++j)
      {
        for(unsigned int k = 0; k < relevant_points_count[j]; ++k)
        {
          AssertIndexRange(relevant_points_offset[j], quadrature_points_count.size());
          auto & qp_counter = quadrature_points_count[relevant_points_offset[j]];
          indices.emplace_back(relevant_points_offset[j], qp_counter);

          ++qp_counter;
        }
      }

      this->indices_per_process[rank] = indices;
    }
  }

  template<typename T>
  void
  process(const std::map<unsigned int, std::vector<std::vector<T>>> & input,
          std::vector<std::vector<T>> &                               output,
          unsigned int const                                          tag) const
  {
    // process remote quadrature points and send them away
    std::map<unsigned int, std::vector<char>> temp_map;

    std::vector<MPI_Request> requests;
    requests.reserve(input.size());

    unsigned int const my_rank = Utilities::MPI::this_mpi_process(comm);

    for(auto const & vec : input)
    {
      if(vec.first == my_rank)
        continue;

      temp_map[vec.first] = Utilities::pack(vec.second);

      auto & buffer = temp_map[vec.first];

      requests.resize(requests.size() + 1);

      MPI_Isend(buffer.data(), buffer.size(), MPI_CHAR, vec.first, tag, comm, &requests.back());
    }

    // receive result

    std::map<unsigned int, std::vector<std::vector<T>>> temp_recv_map;

    // process locally-owned values
    if(input.find(my_rank) != input.end())
      temp_recv_map[my_rank] = input.at(my_rank);

    for(auto const & vec : map_recv)
    {
      if(vec.first == my_rank)
        continue;

      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);

      int message_length;
      MPI_Get_count(&status, MPI_CHAR, &message_length);

      std::vector<char> buffer(message_length);

      MPI_Recv(
        buffer.data(), buffer.size(), MPI_CHAR, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);

      temp_recv_map[status.MPI_SOURCE] = Utilities::unpack<std::vector<std::vector<T>>>(buffer);
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for(auto const & i : temp_recv_map)
    {
      unsigned int const rank = i.first;

      Assert(indices_per_process.find(rank) != indices_per_process.end(),
             ExcMessage("Map does not contain this rank."));

      auto it = indices_per_process.at(rank).begin();

      for(auto const & i : temp_recv_map[rank])
        for(auto const & j : i)
        {
          output[it->first][it->second] = j;
          ++it;
        }
    }
  }

  template<typename T>
  void
  init_solution_values(std::vector<std::vector<T>> & output) const
  {
    output.resize(quadrature_points_count.size());

    for(unsigned int i = 0; i < quadrature_points_count.size(); ++i)
      output[i] = std::vector<T>(quadrature_points_count[i], T());
  }

  template<typename T>
  void
  init_remote_solution_values(std::map<unsigned int, std::vector<std::vector<T>>> & input) const
  {
    for(auto const & i : this->relevant_remote_points_count_per_process)
    {
      unsigned int const rank = i.first;

      std::vector<std::vector<T>> temp(i.second.size());

      for(unsigned int j = 0; j < i.second.size(); ++j)
        temp[j] = std::vector<T>(i.second[j], T());

      input[rank] = temp;
    }
  }

  /**
   * Return quadrature points (sorted according to rank).
   */
  const std::map<unsigned int, std::vector<Point<spacedim>>> &
  get_remote_quadrature_points() const
  {
    return relevant_remote_points_per_process;
  }

private:
  MPI_Comm const comm;

  // receiver side
  std::vector<unsigned int> quadrature_points_count;
  std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>> indices_per_process;
  std::map<unsigned int, std::vector<Point<spacedim>>>                       map_recv;

  // sender side
  std::map<unsigned int, std::vector<Point<spacedim>>> relevant_remote_points_per_process;
  std::map<unsigned int, std::vector<unsigned int>>    relevant_remote_points_count_per_process;
};


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

  typedef std::vector<Point<dim>>                             ArrayQuadraturePoints;
  typedef std::vector<std::vector<Cache>>                     ArrayVectorCache;
  typedef std::vector<std::vector<Tensor<rank, dim, double>>> ArrayVectorTensor;

public:
  InterfaceCoupling() : dof_index_dst(0), dof_handler_src(nullptr), mapping_src(nullptr)
  {
  }

  void
  setup(std::shared_ptr<MatrixFree<dim, Number>>  matrix_free_dst_in,
        unsigned int const                        dof_index_dst_in,
        std::vector<quad_index> const &           quad_indices_dst_in,
        MapBoundaryCondition const &              map_bc_in,
        std::shared_ptr<Triangulation<dim>> const triangulation_src_in,
        DoFHandler<dim> const &                   dof_handler_src_in,
        Mapping<dim> const &                      mapping_src_in,
        VectorType const &                        dof_vector_src_in,
        double const                              tolerance)
  {
    matrix_free_dst   = matrix_free_dst_in;
    dof_index_dst     = dof_index_dst_in;
    quad_rules_dst    = quad_indices_dst_in;
    map_bc            = map_bc_in;
    triangulation_src = triangulation_src_in;
    dof_handler_src   = &dof_handler_src_in;
    mapping_src       = &mapping_src_in;

    std::shared_ptr<GridTools::Cache<dim, dim>> cache_src;
    cache_src.reset(new GridTools::Cache<dim, dim>(*triangulation_src, *mapping_src));

    // mark vertices at interface in order to make search of active cells around point more
    // efficient
    std::vector<bool> marked_vertices(triangulation_src->n_vertices(), false);
    for(auto cell : triangulation_src->active_cell_iterators())
    {
      if(!cell->is_artificial() && cell->at_boundary())
      {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if(cell->face(f)->at_boundary())
          {
            if(map_bc.find(cell->face(f)->boundary_id()) != map_bc.end())
            {
              for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
              {
                marked_vertices[cell->face(f)->vertex_index(v)] = true;
              }
            }
          }
        }
      }
    }

    // implementation needs Number = double
    VectorTypeDouble         dof_vector_src_double_copy;
    VectorTypeDouble const * dof_vector_src_double_ptr;
    if(std::is_same<double, Number>::value)
    {
      dof_vector_src_double_ptr = reinterpret_cast<VectorTypeDouble const *>(&dof_vector_src_in);
    }
    else
    {
      dof_vector_src_double_copy = dof_vector_src_in;
      dof_vector_src_double_ptr  = &dof_vector_src_double_copy;
    }

    for(auto quadrature : quad_rules_dst)
    {
      // initialize maps
      map_index_dst.emplace(quadrature, MapIndex());
      map_q_points_dst.emplace(quadrature, ArrayQuadraturePoints());
      map_solution_dst.emplace(quadrature, ArrayVectorTensor());

      MapIndex &              map_index          = map_index_dst.find(quadrature)->second;
      ArrayQuadraturePoints & array_q_points_dst = map_q_points_dst.find(quadrature)->second;
      ArrayVectorTensor &     array_solution_dst = map_solution_dst.find(quadrature)->second;


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

      map_q_points_src.emplace(quadrature, std::map<mpi_rank, ArrayQuadraturePoints>());
      map_cache_src.emplace(quadrature, std::map<mpi_rank, ArrayVectorCache>());
      map_solution_src.emplace(quadrature, std::map<mpi_rank, ArrayVectorTensor>());

      std::map<mpi_rank, ArrayQuadraturePoints> & mpi_map_q_points_src =
        map_q_points_src.find(quadrature)->second;
      std::map<mpi_rank, ArrayVectorCache> & mpi_map_cache_src =
        map_cache_src.find(quadrature)->second;
      std::map<mpi_rank, ArrayVectorTensor> & mpi_map_solution_src =
        map_solution_src.find(quadrature)->second;


      /*
       * 2. Communication: receive and cache quadrature points of other ranks,
       *    redundantly store own q-points (those that are needed)
       */
      map_communicator.emplace(quadrature,
                               InterfaceCommunicator<dim, dim>(array_q_points_dst,
                                                               *triangulation_src,
                                                               *mapping_src,
                                                               tolerance,
                                                               marked_vertices,
                                                               cache_src));

      InterfaceCommunicator<dim, dim> & communicator = map_communicator.find(quadrature)->second;

      mpi_map_q_points_src = communicator.get_remote_quadrature_points();

      communicator.init_solution_values(array_solution_dst);

      communicator.init_remote_solution_values(mpi_map_solution_src);


      /*
       * 3. Compute dof indices and shape values for all quadrature points
       */
      typename Triangulation<dim, dim>::active_cell_iterator cell_hint =
        triangulation_src->begin_active();

      for(auto it : mpi_map_q_points_src)
      {
        mpi_rank const          proc               = it.first;
        ArrayQuadraturePoints & array_q_points_src = it.second;

        mpi_map_cache_src.emplace(proc, ArrayVectorCache());
        ArrayVectorCache & array_cache_src = mpi_map_cache_src.find(proc)->second;
        array_cache_src.resize(array_q_points_src.size());

        for(types::global_dof_index q = 0; q < array_q_points_src.size(); ++q)
        {
          auto adjacent_cells =
            find_all_active_cells_around_point(*mapping_src,
                                               dof_handler_src->get_triangulation(),
                                               array_q_points_src[q],
                                               tolerance,
                                               cell_hint,
                                               marked_vertices,
                                               *cache_src);

          array_cache_src[q] = get_dof_indices_and_shape_values(adjacent_cells,
                                                                *dof_handler_src,
                                                                *mapping_src,
                                                                *dof_vector_src_double_ptr);

          AssertThrow(array_cache_src[q].size() > 0,
                      ExcMessage("No adjacent points have been found."));
        }
      }


      /*
       * 4. Communication: transfer results back to dst-side
       */
      communicator.process(mpi_map_solution_src, array_solution_dst, 11 + quadrature);
    }

    // finally, give boundary condition access to the data
    for(auto boundary : map_bc)
    {
      boundary.second->set_data_pointer(map_index_dst, map_solution_dst);
    }
  }

  void
  update_data(VectorType const & dof_vector_src)
  {
    VectorTypeDouble         dof_vector_src_double_copy;
    VectorTypeDouble const * dof_vector_src_double_ptr;
    if(std::is_same<double, Number>::value)
    {
      dof_vector_src_double_ptr = reinterpret_cast<VectorTypeDouble const *>(&dof_vector_src);
    }
    else
    {
      dof_vector_src_double_copy = dof_vector_src;
      dof_vector_src_double_ptr  = &dof_vector_src_double_copy;
    }

    dof_vector_src_double_ptr->update_ghost_values();

    /*
     * 1. Interpolate data on src-side
     */
    for(auto quadrature : quad_rules_dst)
    {
      ArrayVectorTensor & array_solution_dst = map_solution_dst.find(quadrature)->second;

      std::map<mpi_rank, ArrayVectorCache> & mpi_map_cache_src =
        map_cache_src.find(quadrature)->second;
      std::map<mpi_rank, ArrayVectorTensor> & mpi_map_solution_src =
        map_solution_src.find(quadrature)->second;

      for(auto it : mpi_map_cache_src)
      {
        mpi_rank const      proc               = it.first;
        ArrayVectorCache &  array_cache_src    = mpi_map_cache_src.find(proc)->second;
        ArrayVectorTensor & array_solution_src = mpi_map_solution_src.find(proc)->second;

        for(types::global_dof_index q = 0; q < array_cache_src.size(); ++q)
        {
          std::vector<Cache> &                     vector_cache    = array_cache_src[q];
          std::vector<Tensor<rank, dim, double>> & vector_solution = array_solution_src[q];

          // interpolate solution from dof vector using cached data
          for(unsigned int i = 0; i < vector_cache.size(); ++i)
          {
            vector_solution[i] = Interpolator<rank, dim, double>::value(*dof_handler_src,
                                                                        *dof_vector_src_double_ptr,
                                                                        vector_cache[i].first,
                                                                        vector_cache[i].second);
          }
        }
      }

      /*
       * 2. Communication: transfer results back to dst-side
       */
      InterfaceCommunicator<dim, dim> & communicator = map_communicator.find(quadrature)->second;

      communicator.process(mpi_map_solution_src, array_solution_dst, 11 + quadrature);
    }
  }

private:
  /*
   * dst-side
   */
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst;
  unsigned int                             dof_index_dst;
  std::vector<quad_index>                  quad_rules_dst;

  mutable std::map<quad_index, MapIndex>              map_index_dst;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points_dst;
  mutable std::map<quad_index, ArrayVectorTensor>     map_solution_dst;

  std::map<quad_index, InterfaceCommunicator<dim, dim>> map_communicator;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionCached<rank, dim, double>>> map_bc;

  /*
   * src-side
   */
  std::shared_ptr<Triangulation<dim>> triangulation_src;
  DoFHandler<dim> const *             dof_handler_src;
  Mapping<dim> const *                mapping_src;

  mutable std::map<quad_index, std::map<mpi_rank, ArrayQuadraturePoints>> map_q_points_src;
  mutable std::map<quad_index, std::map<mpi_rank, ArrayVectorCache>>      map_cache_src;
  mutable std::map<quad_index, std::map<mpi_rank, ArrayVectorTensor>>     map_solution_src;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
