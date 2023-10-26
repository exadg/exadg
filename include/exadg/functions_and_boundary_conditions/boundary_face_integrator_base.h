/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_BOUNDARY_FACE_INTEGRATOR_BASE_H_
#define EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_BOUNDARY_FACE_INTEGRATOR_BASE_H_

#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
/**
 * Base class to access boundary values with a similar interface to FaceIntegrators.
 * In the derived class a @c get_value() has to be implemented. Since the return type
 * depends on the derived class, a pure virtual function for this purpose can not be added.
 */
template<typename BoundaryDescriptorType, typename Number>
class BoundaryFaceIntegratorBase
{
  using BoundaryType       = typename BoundaryDescriptorType::BoundaryType;
  static constexpr int dim = BoundaryDescriptorType::dimension;

public:
  void
  reinit(unsigned int const face, Number const time)
  {
    evaluation_time = time;

    auto const boundary_id_new = matrix_free.get_boundary_id(face);

    // only update boundary_type if needed to avoid an unnecessary search in boundary_descriptor
    if(boundary_id_new != boundary_id)
    {
      boundary_id   = boundary_id_new;
      boundary_type = boundary_descriptor.get_boundary_type(boundary_id);
    }
  }

  // A corresponding function has to be implemented in the deriving class.
  // The return type varies dependent on the value type.
  // auto
  // get_value();

protected:
  BoundaryFaceIntegratorBase(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                             BoundaryDescriptorType const &          boundary_descriptor_in)
    : matrix_free(matrix_free_in),
      boundary_descriptor(boundary_descriptor_in),
      evaluation_time(Number{0.0}),
      boundary_id(dealii::numbers::invalid_boundary_id),
      boundary_type(Utilities::default_constructor<BoundaryType>())
  {
  }

  dealii::MatrixFree<dim, Number> const & matrix_free;
  BoundaryDescriptorType const &          boundary_descriptor;

  Number                     evaluation_time;
  dealii::types::boundary_id boundary_id;
  BoundaryType               boundary_type;
};

} // namespace ExaDG

#endif /*EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_BOUNDARY_FACE_INTEGRATOR_BASE_H_*/
