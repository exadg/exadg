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

#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>

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

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(
      dealii::types::boundary_id const             boundary_id,
      std::set<dealii::types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
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

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(
      dealii::types::boundary_id const             boundary_id,
      std::set<dealii::types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

template<int dim>
struct BoundaryDescriptor
{
  BoundaryDescriptor()
  {
    pressure = std::make_shared<BoundaryDescriptorP<dim>>();
    velocity = std::make_shared<BoundaryDescriptorU<dim>>();
  }

  std::shared_ptr<BoundaryDescriptorP<dim>> pressure;
  std::shared_ptr<BoundaryDescriptorU<dim>> velocity;
};

template<int dim>
inline void
verify_boundary_conditions(BoundaryDescriptor<dim> const & boundary_descriptor,
                           Grid<dim> const &               grid)
{
  ExaDG::verify_boundary_conditions(*boundary_descriptor.pressure, grid);
  ExaDG::verify_boundary_conditions(*boundary_descriptor.velocity, grid);
}

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
