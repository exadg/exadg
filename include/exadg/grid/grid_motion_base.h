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

#ifndef INCLUDE_EXADG_GRID_GRID_MOTION_BASE_H_
#define INCLUDE_EXADG_GRID_GRID_MOTION_BASE_H_

// ExaDG
#include <exadg/grid/grid_motion_interface.h>
#include <exadg/grid/mapping_dof_vector.h>

namespace ExaDG
{
using namespace dealii;

/**
 * Base class for moving grid problems.
 */
template<int dim, typename Number>
class GridMotionBase : public GridMotionInterface<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  GridMotionBase(std::shared_ptr<Mapping<dim> const> mapping_undeformed,
                 unsigned int const                  mapping_degree_q_cache,
                 Triangulation<dim> const &          triangulation)
    : mapping_undeformed(mapping_undeformed)
  {
    // Make sure that MappingQCache is initialized correctly. An empty dof-vector is used and,
    // hence, no displacements are added to the reference configuration described by
    // mapping_undeformed.
    DoFHandler<dim> dof_handler(triangulation);
    VectorType      displacement_vector;
    moving_mapping = std::make_shared<MappingDoFVector<dim, Number>>(mapping_degree_q_cache);
    moving_mapping->initialize_mapping_q_cache(mapping_undeformed,
                                               displacement_vector,
                                               dof_handler);
  }

  void
  fill_grid_coordinates_vector(VectorType &            grid_coordinates,
                               DoFHandler<dim> const & dof_handler) const final
  {
    moving_mapping->fill_grid_coordinates_vector(grid_coordinates, dof_handler);
  }

  std::shared_ptr<Mapping<dim> const>
  get_mapping() const final
  {
    return moving_mapping;
  }

protected:
  // mapping describing undeformed reference state
  std::shared_ptr<Mapping<dim> const> mapping_undeformed;

  // time-dependent mapping describing deformed state
  std::shared_ptr<MappingDoFVector<dim, Number>> moving_mapping;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_GRID_MOTION_BASE_H_ */
