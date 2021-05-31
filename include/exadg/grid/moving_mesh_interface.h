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

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/la_parallel_vector.h>

#ifndef INCLUDE_EXADG_GRID_MOVING_MESH_INTERFACE_H_
#  define INCLUDE_EXADG_GRID_MOVING_MESH_INTERFACE_H_

namespace ExaDG
{
using namespace dealii;

/**
 * Pure-virtual interface class for moving mesh functionality.
 */
template<int dim, typename Number>
class MovingMeshInterface
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Destructor.
   */
  virtual ~MovingMeshInterface()
  {
  }

  /**
   * Updates the mapping, i.e., moves the mesh.
   */
  virtual void
  update(double const time, bool const print_solver_info) = 0;

  /**
   * Print the number of iterations for PDE type mesh motion problems.
   */
  virtual void
  print_iterations() const
  {
    AssertThrow(false, ExcMessage("Has to be overwritten by derived classes."));
  }

  /**
   * Extract the grid coordinates of the current mesh configuration and fill a dof-vector given a
   * corresponding DoFHandler object.
   */
  virtual void
  fill_grid_coordinates_vector(VectorType &            grid_coordinates,
                               DoFHandler<dim> const & dof_handler) const = 0;

  /**
   * Return a shared pointer to dealii::Mapping<dim>.
   */
  virtual std::shared_ptr<Mapping<dim>>
  get_mapping() = 0;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_MOVING_MESH_INTERFACE_H_ */
