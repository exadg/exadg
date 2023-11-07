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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

#include <set>

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

namespace ExaDG
{
namespace Acoustics
{
enum class BoundaryTypeP
{
  Undefined,
  Dirichlet,
};

enum class BoundaryTypeU
{
  Undefined,
  Neumann
};

template<int dim>
struct BoundaryDescriptorP
{
  static constexpr int dimension = dim;
  using BoundaryType             = BoundaryTypeP;

  // Dirichlet: prescribe pressure
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeP
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryTypeP::Dirichlet;

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryTypeP::Undefined;
  }
};

template<int dim>
struct BoundaryDescriptorU
{
  static constexpr int dimension = dim;
  using BoundaryType             = BoundaryTypeU;

  // Neumann: only the boundary IDs are stored but no inhomogeneous boundary conditions are
  // prescribed
  std::set<dealii::types::boundary_id> neumann_bc;

  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeU
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryTypeU::Neumann;

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryTypeU::Undefined;
  }
};

template<int dim>
struct BoundaryDescriptor
{
  std::shared_ptr<BoundaryDescriptorP<dim>> pressure;
  std::shared_ptr<BoundaryDescriptorU<dim>> velocity;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
