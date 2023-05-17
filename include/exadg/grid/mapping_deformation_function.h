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

#ifndef INCLUDE_GRID_MAPPING_DEFORMATION_FUNCTION_H_
#define INCLUDE_GRID_MAPPING_DEFORMATION_FUNCTION_H_

#include <exadg/grid/mapping_deformation_base.h>

namespace ExaDG
{
/**
 * Class for mesh deformations that can be described analytically via a dealii::Function<dim>
 * object.
 */
template<int dim, typename Number>
class DeformedMappingFunction : public DeformedMappingBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  DeformedMappingFunction(std::shared_ptr<dealii::Mapping<dim> const>  mapping_undeformed,
                          unsigned int const                           mapping_degree_q_cache,
                          dealii::Triangulation<dim> const &           triangulation,
                          std::shared_ptr<dealii::Function<dim>> const mesh_deformation_function,
                          double const                                 start_time)
    : DeformedMappingBase<dim, Number>(mapping_undeformed, mapping_degree_q_cache, triangulation),
      mesh_deformation_function(mesh_deformation_function),
      triangulation(triangulation)
  {
    update(start_time, false, dealii::numbers::invalid_unsigned_int);
  }

  /**
   * Updates the grid coordinates using a dealii::Function<dim> object evaluated at a given time.
   */
  void
  update(double const     time,
         bool const       print_solver_info,
         types::time_step time_step_number) override
  {
    (void)print_solver_info;
    (void)time_step_number;

    mesh_deformation_function->set_time(time);

    this->do_initialize(triangulation, mesh_deformation_function);
  }

private:
  /**
   * Initializes the dealii::MappingQCache object by providing a dealii::Function<dim> that
   * describes the displacement of the grid compared to an undeformed reference configuration
   * described by the mapping_undeformed member of the base class.
   */
  void
  do_initialize(dealii::Triangulation<dim> const &     triangulation,
                std::shared_ptr<dealii::Function<dim>> displacement_function)
  {
    AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

    // dummy FE for compatibility with interface of dealii::FEValues
    dealii::FE_Nothing<dim> dummy_fe;
    dealii::FEValues<dim>   fe_values(*this->mapping_undeformed,
                                    dummy_fe,
                                    dealii::QGaussLobatto<dim>(this->get_degree() + 1),
                                    dealii::update_quadrature_points);

    this->initialize(triangulation,
                     [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
                       -> std::vector<dealii::Point<dim>> {
                       fe_values.reinit(cell);

                       // compute displacement and add to original position
                       std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
                       for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
                       {
                         // need to adjust for hierarchic numbering of dealii::MappingQCache
                         dealii::Point<dim> const point = fe_values.quadrature_point(
                           this->hierarchic_to_lexicographic_numbering[i]);
                         dealii::Point<dim> displacement;
                         for(unsigned int d = 0; d < dim; ++d)
                           displacement[d] = displacement_function->value(point, d);

                         points_moved[i] = point + displacement;
                       }

                       return points_moved;
                     });
  }

  std::shared_ptr<dealii::Function<dim>> mesh_deformation_function;

  dealii::Triangulation<dim> const & triangulation;
};

} // namespace ExaDG

#endif /*INCLUDE_GRID_MAPPING_DEFORMATION_FUNCTION_H_*/
