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

#ifndef INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_BASE_H_
#define INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_BASE_H_

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
 * Base class to describe grid deformations, where the grid coordinates described by a reference
 * mapping are superimposed by a displacement DoF vector.
 *
 * A typical use case of this class are Arbitrary Lagrangian-Eulerian type problems with moving
 * domains / grids, where the displacement vector describes the time-dependent deformation of the
 * grid. However, this class may also be used for stationary problems, e.g. to apply high-order
 * curved boundaries on top of low-order approximation of a certain geometry.
 */
template<int dim, typename Number>
class DeformedMappingBase : public MappingDoFVector<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   *
   * The constructor assumes a vanishing displacement field, i.e. after construction
   * the reference configuration described by @p mapping_undeformed describes the
   * absolute coordinates of the grid.
   */
  DeformedMappingBase(std::shared_ptr<dealii::Mapping<dim> const> mapping_undeformed,
                      dealii::Triangulation<dim> const &          triangulation)
    : MappingDoFVector<dim, Number>(triangulation), mapping_undeformed(mapping_undeformed)
  {
  }

  /**
   * Desctructor.
   */
  virtual ~DeformedMappingBase()
  {
  }

  /**
   * Updates the mapping, i.e., moves the grid.
   *
   * TODO: The parameters print_solver_info and time_step_number are only relevant for PDE-type grid
   * deformation problems. Hence this function with these particular arguments should not appear in
   * this base class.
   */
  virtual void
  update(double const time, bool const print_solver_info, types::time_step time_step_number) = 0;

  /**
   * Print the number of iterations for PDE-type grid deformation problems.
   *
   * TODO: this function is only relevant for PDE-type grid deformation problems, and should
   * therefore not appear in this base class.
   */
  virtual void
  print_iterations() const
  {
    AssertThrow(false, dealii::ExcMessage("Has to be overwritten by derived classes."));
  }

  /**
   * Calls corresponding function of base class MappingDoFVector using the member variable
   * mapping_undeformed.
   */
  void
  initialize_mapping_from_function(dealii::Triangulation<dim> const &     triangulation,
                                   unsigned int const                     mapping_degree,
                                   std::shared_ptr<dealii::Function<dim>> displacement_function)
  {
    MappingDoFVector<dim, Number>::initialize_mapping_from_function(mapping_undeformed,
                                                                    triangulation,
                                                                    mapping_degree,
                                                                    displacement_function);
  }

  /**
   * Calls corresponding function of base class MappingDoFVector using the member variable
   * mapping_undeformed.
   */
  void
  initialize_mapping_from_dof_vector(VectorType const &              displacement_vector,
                                     dealii::DoFHandler<dim> const & dof_handler)
  {
    MappingDoFVector<dim, Number>::initialize_mapping_from_dof_vector(mapping_undeformed,
                                                                      displacement_vector,
                                                                      dof_handler);
  }

private:
  /**
   * mapping describing undeformed reference state
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping_undeformed;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_BASE_H_ */
