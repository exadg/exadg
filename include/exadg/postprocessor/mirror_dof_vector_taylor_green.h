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

#ifndef INCLUDE_FUNCTIONALITIES_MIRROR_DOF_VECTOR_TAYLOR_GREEN_H_
#define INCLUDE_FUNCTIONALITIES_MIRROR_DOF_VECTOR_TAYLOR_GREEN_H_

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/*
 * Taylor-Green symmetries:
 *
 *  v_i (..., - x_j,..., t) = - v_i(..., x_j,..., t) if i = j
 *
 * and
 *
 *  v_i (..., - x_j,..., t) = + v_i(..., x_j,..., t) if i != j
 */
template<int dim, typename Number>
void
apply_taylor_green_symmetry(dealii::DoFHandler<dim> const & dof_handler_symm,
                            dealii::DoFHandler<dim> const & dof_handler,
                            double const                    n_cells_1d,
                            double const                    delta,
                            const dealii::LinearAlgebra::distributed::Vector<Number> & vector_symm,
                            dealii::LinearAlgebra::distributed::Vector<Number> &       vector)
{
  // determine some useful constants
  auto const & fe = dof_handler.get_fe();

  MPI_Comm const comm = dof_handler.get_communicator();

  // determine which process has which index (lex numbering) and wants which
  dealii::IndexSet range_has_lex(dof_handler_symm.n_dofs());  // has in symm system
  dealii::IndexSet range_want_lex(dof_handler_symm.n_dofs()); // want in full system

  // ... and create a map: lex to cell iterators
  std::map<unsigned int, typename dealii::DoFHandler<dim>::active_cell_iterator>
    map_lex_to_cell_symm;
  std::map<unsigned int, std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
    map_lex_to_cell_full;

  {
    auto norm_point_to_lex = [&](dealii::Point<dim> const c) {
      // convert normalized point [0, 1] to lex
      if(dim == 2)
        return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1d * std::floor(c[1]));
      else
        return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1d * std::floor(c[1]) +
                                        n_cells_1d * n_cells_1d * std::floor(c[2]));
    };

    // ... has (symm)
    for(auto const & cell : dof_handler_symm.active_cell_iterators())
      if(cell->is_active() && cell->is_locally_owned())
      {
        auto c = cell->center();
        for(unsigned int i = 0; i < dim; i++)
          c[i] = c[i] / delta;

        unsigned int const lid = norm_point_to_lex(c);

        range_has_lex.add_index(lid);
        map_lex_to_cell_symm[lid] = cell;
      }

    // want (full)
    for(auto const & cell : dof_handler.active_cell_iterators())
      if(cell->is_active() && cell->is_locally_owned())
      {
        auto c = cell->center();
        for(unsigned int i = 0; i < dim; i++)
          c[i] = std::abs(c[i]) / delta;

        unsigned int const lex = norm_point_to_lex(c);

        range_want_lex.add_index(lex);
        map_lex_to_cell_full[lex].emplace_back(cell);
      }
  }

  // determine who has and who wants data
  std::map<unsigned int, std::vector<unsigned int>> recv_map_proc_to_lex_offset;
  std::map<unsigned int, dealii::IndexSet>          send_map_proc_to_lex;

  {
    std::vector<unsigned int> owning_ranks_of_ghosts(range_want_lex.n_elements());

    // set up dictionary
    dealii::Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload process(
      range_has_lex, range_want_lex, comm, owning_ranks_of_ghosts, true);

    dealii::Utilities::MPI::ConsensusAlgorithms::Selector<
      std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>,
      unsigned int>
#if DEAL_II_VERSION_GTE(10, 0, 0)
      consensus_algorithm;
    consensus_algorithm.run(process, comm);
#else
      consensus_algorithm(process, comm);
    consensus_algorithm.run();
#endif

    for(auto const & owner : owning_ranks_of_ghosts)
      recv_map_proc_to_lex_offset[owner] = std::vector<unsigned int>();

    for(unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
      recv_map_proc_to_lex_offset[owning_ranks_of_ghosts[i]].push_back(i);

    send_map_proc_to_lex = process.get_requesters();
  }

  // perform data exchange and fill this buffer
  std::vector<double> data_buffer(range_want_lex.n_elements() * fe.n_dofs_per_cell());
  {
    // data structure for MPI exchange
    std::map<unsigned int, std::vector<double>> recv_buffer;
    {
      std::map<unsigned int, std::vector<double>> send_buffers;

      std::vector<MPI_Request> recv_requests(recv_map_proc_to_lex_offset.size());
      std::vector<MPI_Request> send_requests(send_map_proc_to_lex.size());

      unsigned int recv_couter = 0;
      unsigned int send_couter = 0;

      // post recv
      for(auto const & recv_offset : recv_map_proc_to_lex_offset)
      {
        recv_buffer[recv_offset.first].resize(recv_offset.second.size() * fe.n_dofs_per_cell());
        MPI_Irecv(recv_buffer[recv_offset.first].data(),
                  recv_buffer[recv_offset.first].size(),
                  MPI_DOUBLE,
                  recv_offset.first,
                  0,
                  comm,
                  &recv_requests[recv_couter++]);
      }

      // post send
      for(auto const & send_index_set : send_map_proc_to_lex)
      {
        // allocate memory
        auto & send_buffer = send_buffers[send_index_set.first];
        send_buffer.resize(send_index_set.second.n_elements() * fe.n_dofs_per_cell());

        // collect data to be send
        auto                                         send_buffer_ptr = &send_buffer[0];
        std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        for(auto const cell_index : send_index_set.second)
        {
          auto const & cell_accessor = map_lex_to_cell_symm[cell_index];
          cell_accessor->get_dof_indices(dof_indices);

          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++)
            send_buffer_ptr[i] = vector_symm[dof_indices[i]];
          send_buffer_ptr += fe.n_dofs_per_cell();
        }

        // send data
        MPI_Isend(send_buffer.data(),
                  send_buffer.size(),
                  MPI_DOUBLE,
                  send_index_set.first,
                  0,
                  comm,
                  &send_requests[send_couter++]);
      }

      // wait that data has been send and received
      MPI_Waitall(recv_couter, recv_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(send_couter, send_requests.data(), MPI_STATUSES_IGNORE);

      // copy received data into a single buffer
      for(auto const & recv_offset : recv_map_proc_to_lex_offset)
      {
        auto const & buffer = recv_buffer[recv_offset.first];

        unsigned int counter = 0;
        for(auto const & offset : recv_offset.second)
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++)
            data_buffer[offset * fe.n_dofs_per_cell() + i] = buffer[counter++];
      }
    }
  }

  // read buffer and fill full vector
  {
    auto send_buffer_ptr = &data_buffer[0];

    unsigned int const n_dofs_per_component = fe.n_dofs_per_cell() / dim;

    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for(auto const cell_index : range_want_lex)
    {
      auto const & cell_accessors = map_lex_to_cell_full[cell_index];

      for(auto const & cell_accessor : cell_accessors)
      {
        cell_accessor->get_dof_indices(dof_indices);

        for(unsigned int i = 0; i < n_dofs_per_component; i++)
        {
          dealii::Point<dim, unsigned int> p =
            dim == 2 ? dealii::Point<dim, unsigned int>(i % (fe.degree + 1), i / (fe.degree + 1)) :
                       dealii::Point<dim, unsigned int>(i % (fe.degree + 1),
                                                        (i % ((fe.degree + 1) * (fe.degree + 1))) /
                                                          (fe.degree + 1),
                                                        i / (fe.degree + 1) / (fe.degree + 1));

          auto c = cell_accessor->center();

          for(unsigned int v = 0; v < dim; v++)
            if(c[v] < 0)
              p[v] = fe.degree - p[v];

          unsigned int const shift =
            dim == 2 ? (p[0] + p[1] * (1 + fe.degree)) :
                       (p[0] + p[1] * (1 + fe.degree) + p[2] * (1 + fe.degree) * (1 + fe.degree));

          for(unsigned int d = 0; d < dim; d++)
            vector[dof_indices[i + d * n_dofs_per_component]] =
              send_buffer_ptr[shift + d * n_dofs_per_component] * (c[d] < 0.0 ? -1.0 : +1.0);
        }
      }
      send_buffer_ptr += fe.n_dofs_per_cell();
    }
  }
}

template<typename MeshType, typename Number>
void
initialize_dof_vector(dealii::LinearAlgebra::distributed::Vector<Number> & vec,
                      const MeshType &                                     dof_handler)
{
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  MPI_Comm const comm = dof_handler.get_communicator();

  vec.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, comm);
}

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MIRROR_DOF_VECTOR_TAYLOR_GREEN_H_ */
