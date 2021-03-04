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

/**
 * A mapping class based on MappingQCache equipped with practical interfaces that can be used to
 * initialize the mapping by providing a Function<dim> object.
 */
template<int dim, typename Number>
class MovingMeshFunction : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MovingMeshFunction(std::shared_ptr<Mapping<dim>>            mapping,
                     unsigned int const                       mapping_degree_q_cache,
                     parallel::TriangulationBase<dim> const & triangulation,
                     std::shared_ptr<Function<dim>> const     mesh_movement_function,
                     double const                             start_time)
    : MovingMeshBase<dim, Number>(mapping, mapping_degree_q_cache, triangulation),
      mesh_movement_function(mesh_movement_function),
      triangulation(triangulation)
  {
    update(start_time, false, false);
  }

  /**
   * Updates the mesh coordinates using a Function<dim> object evaluated at a given time.
   */
  void
  update(double const time, bool const print_solver_info, bool const print_wall_times) override
  {
    (void)print_solver_info;
    (void)print_wall_times;

    mesh_movement_function->set_time(time);

    this->initialize(triangulation, mesh_movement_function);
  }

private:
  std::shared_ptr<Function<dim>> mesh_movement_function;

  Triangulation<dim> const & triangulation;
};

} // namespace ExaDG

#endif /*INCLUDE_MOVING_MESH_H_*/
