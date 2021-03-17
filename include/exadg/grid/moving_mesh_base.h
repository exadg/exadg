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

#ifndef INCLUDE_EXADG_GRID_MOVING_MESH_BASE_H_
#define INCLUDE_EXADG_GRID_MOVING_MESH_BASE_H_

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>

namespace ExaDG
{
using namespace dealii;

/**
 * Base class for moving mesh problems based on MappingFiniteElement.
 */
template<int dim, typename Number>
class MovingMeshBase : public MappingDoFVector<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MovingMeshBase(std::shared_ptr<Mapping<dim> const> mapping_undeformed,
                 unsigned int const                  mapping_degree_q_cache,
                 Triangulation<dim> const &          triangulation)
    : MappingDoFVector<dim, Number>(mapping_degree_q_cache), mapping_undeformed(mapping_undeformed)
  {
    // Make sure that MappingQCache is initialized correctly. An empty dof-vector is used and,
    // hence, no displacements are added to the reference configuration described by
    // mapping_undeformed.
    DoFHandler<dim> dof_handler(triangulation);
    VectorType      displacement_vector;
    this->initialize_mapping_q_cache(mapping_undeformed, displacement_vector, dof_handler);
  }

  /**
   * Destructor.
   */
  virtual ~MovingMeshBase()
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

protected:
  // mapping describing undeformed reference state
  std::shared_ptr<Mapping<dim> const> mapping_undeformed;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MOVING_MESH_BASE_H_ */
