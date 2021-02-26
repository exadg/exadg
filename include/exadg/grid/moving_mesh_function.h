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

#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <exadg/grid/moving_mesh_base.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MovingMeshFunction : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshFunction(parallel::TriangulationBase<dim> const & triangulation_in,
                     std::shared_ptr<Mapping<dim>>            mapping_in,
                     unsigned int const                       mapping_degree_moving_in,
                     MPI_Comm const &                         mpi_comm_in,
                     std::shared_ptr<Function<dim>> const     mesh_movement_function_in,
                     double const                             start_time)
    : MovingMeshBase<dim, Number>(mapping_in, mapping_degree_moving_in, mpi_comm_in),
      mesh_movement_function(mesh_movement_function_in),
      triangulation(triangulation_in)
  {
    move_mesh(start_time);
  }

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh(double const time, bool const print_solver_info = false)
  {
    (void)print_solver_info;

    mesh_movement_function->set_time(time);

    this->initialize_mapping_q_cache(triangulation, mesh_movement_function);
  }

private:
  std::shared_ptr<Function<dim>> mesh_movement_function;

  Triangulation<dim> const & triangulation;
};

} // namespace ExaDG

#endif /*INCLUDE_MOVING_MESH_H_*/
