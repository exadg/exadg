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

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
/**
 * Base class for moving grid problems.
 */
template<int dim, typename Number>
class GridMotionBase : public MappingDoFVector<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  GridMotionBase(std::shared_ptr<dealii::Mapping<dim> const> mapping_undeformed,
                 unsigned int const                          mapping_degree_q_cache,
                 dealii::Triangulation<dim> const &          triangulation)
    : MappingDoFVector<dim, Number>(mapping_degree_q_cache), mapping_undeformed(mapping_undeformed)
  {
    // Make sure that dealii::MappingQCache is initialized correctly. An empty dof-vector is used
    // and, hence, no displacements are added to the reference configuration described by
    // mapping_undeformed.
    dealii::DoFHandler<dim> dof_handler(triangulation);
    VectorType              displacement_vector;
    this->initialize_mapping_q_cache(mapping_undeformed, displacement_vector, dof_handler);
  }

  virtual ~GridMotionBase()
  {
  }

  /**
   * Updates the mapping, i.e., moves the grid.
   */
  virtual void
  update(double const time, bool const print_solver_info, types::time_step time_step_number) = 0;

  /**
   * Print the number of iterations for PDE type grid motion problems.
   */
  virtual void
  print_iterations() const
  {
    AssertThrow(false, dealii::ExcMessage("Has to be overwritten by derived classes."));
  }

protected:
  // mapping describing undeformed reference state
  std::shared_ptr<dealii::Mapping<dim> const> mapping_undeformed;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_GRID_MOTION_BASE_H_ */
