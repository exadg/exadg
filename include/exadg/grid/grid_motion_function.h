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

#ifndef INCLUDE_GRID_GRID_MOTION_ANALYTICAL_H_
#define INCLUDE_GRID_GRID_MOTION_ANALYTICAL_H_

#include <exadg/grid/grid_motion_base.h>

namespace ExaDG
{
/**
 * Class for moving mesh problems based on mesh motions that can be described analytically via a
 * dealii::Function<dim> object.
 */
template<int dim, typename Number>
class GridMotionFunction : public GridMotionBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  GridMotionFunction(std::shared_ptr<dealii::Mapping<dim> const>  mapping_undeformed,
                     unsigned int const                           mapping_degree_q_cache,
                     dealii::Triangulation<dim> const &           triangulation,
                     std::shared_ptr<dealii::Function<dim>> const mesh_movement_function,
                     double const                                 start_time)
    : GridMotionBase<dim, Number>(mapping_undeformed, mapping_degree_q_cache, triangulation),
      mesh_movement_function(mesh_movement_function),
      triangulation(triangulation)
  {
    update(start_time, false);
  }

  /**
   * Updates the grid coordinates using a dealii::Function<dim> object evaluated at a given time.
   */
  void
  update(double const time, bool const print_solver_info) override
  {
    (void)print_solver_info;

    mesh_movement_function->set_time(time);

    this->initialize(triangulation, mesh_movement_function);
  }

private:
  /**
   * Initializes the dealii::MappingQCache object by providing a dealii::Function<dim> that
   * describes the displacement of the grid compared to an undeformed reference configuration
   * described by the static mapping of this class.
   */
  void
  initialize(dealii::Triangulation<dim> const &     triangulation,
             std::shared_ptr<dealii::Function<dim>> displacement_function)
  {
    AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

    // dummy FE for compatibility with interface of dealii::FEValues
    dealii::FE_Nothing<dim> dummy_fe;
    dealii::FEValues<dim>   fe_values(*this->mapping_undeformed,
                                    dummy_fe,
                                    dealii::QGaussLobatto<dim>(this->moving_mapping->get_degree() +
                                                               1),
                                    dealii::update_quadrature_points);

    this->moving_mapping->initialize(
      triangulation,
      [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
        -> std::vector<dealii::Point<dim>> {
        fe_values.reinit(cell);

        // compute displacement and add to original position
        std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          // need to adjust for hierarchic numbering of dealii::MappingQCache
          dealii::Point<dim> const point = fe_values.quadrature_point(
            this->moving_mapping->hierarchic_to_lexicographic_numbering[i]);
          dealii::Point<dim> displacement;
          for(unsigned int d = 0; d < dim; ++d)
            displacement[d] = displacement_function->value(point, d);

          points_moved[i] = point + displacement;
        }

        return points_moved;
      });
  }

  std::shared_ptr<dealii::Function<dim>> mesh_movement_function;

  dealii::Triangulation<dim> const & triangulation;
};

} // namespace ExaDG

#endif /*INCLUDE_GRID_GRID_MOTION_ANALYTICAL_H_*/
