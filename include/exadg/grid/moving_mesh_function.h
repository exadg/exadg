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
 * Class for moving mesh problems based on mesh motions that can be described analytically via a
 * Function<dim> object.
 */
template<int dim, typename Number>
class MovingMeshFunction : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MovingMeshFunction(std::shared_ptr<Mapping<dim> const>  mapping_undeformed,
                     unsigned int const                   mapping_degree_q_cache,
                     Triangulation<dim> const &           triangulation,
                     std::shared_ptr<Function<dim>> const mesh_movement_function,
                     double const                         start_time)
    : MovingMeshBase<dim, Number>(mapping_undeformed, mapping_degree_q_cache, triangulation),
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

  /**
   * Initializes the MappingQCache object by providing a Function<dim> that describes the
   * displacement of the mesh compared to an undeformed reference configuration described by the
   * static mapping of this class.
   */
  void
  initialize(Triangulation<dim> const &     triangulation,
             std::shared_ptr<Function<dim>> displacement_function)
  {
    AssertThrow(MultithreadInfo::n_threads() == 1, ExcNotImplemented());

    // dummy FE for compatibility with interface of FEValues
    FE_Nothing<dim> dummy_fe;
    FEValues<dim>   fe_values(*this->mapping_undeformed,
                            dummy_fe,
                            QGaussLobatto<dim>(this->get_degree() + 1),
                            update_quadrature_points);

    MappingQCache<dim>::initialize(
      triangulation,
      [&](typename Triangulation<dim>::cell_iterator const & cell) -> std::vector<Point<dim>> {
        fe_values.reinit(cell);

        // compute displacement and add to original position
        std::vector<Point<dim>> points_moved(fe_values.n_quadrature_points);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          // need to adjust for hierarchic numbering of MappingQCache
          Point<dim> const point =
            fe_values.quadrature_point(this->hierarchic_to_lexicographic_numbering[i]);
          Point<dim> displacement;
          for(unsigned int d = 0; d < dim; ++d)
            displacement[d] = displacement_function->value(point, d);

          points_moved[i] = point + displacement;
        }

        return points_moved;
      });
  }

private:
  std::shared_ptr<Function<dim>> mesh_movement_function;

  Triangulation<dim> const & triangulation;
};

} // namespace ExaDG

#endif /*INCLUDE_MOVING_MESH_H_*/
