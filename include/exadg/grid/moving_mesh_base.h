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
#include <exadg/grid/mapping_finite_element.h>

namespace ExaDG
{
using namespace dealii;

/**
 * Base class for moving mesh problems based on MappingFiniteElement.
 */
template<int dim, typename Number>
class MovingMeshBase : public MappingFiniteElement<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MovingMeshBase(std::shared_ptr<Mapping<dim>> mapping,
                 unsigned int const            mapping_degree_q_cache,
                 Triangulation<dim> const &    triangulation,
                 MPI_Comm const &              mpi_comm)
    : MappingFiniteElement<dim, Number>(mapping, mapping_degree_q_cache, triangulation, mpi_comm)
  {
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
  update(double const time, bool const print_solver_info = false) = 0;

  /**
   * Print the number of iterations for PDE type mesh motion problems.
   */
  virtual void
  print_iterations() const
  {
    AssertThrow(false, ExcMessage("Has to be overwritten by derived classes."));
  }
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MOVING_MESH_BASE_H_ */
