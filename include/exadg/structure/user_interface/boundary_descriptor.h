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

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/component_mask.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>

namespace ExaDG
{
namespace Structure
{
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  DirichletCached,
  Neumann,
  NeumannCached,
  RobinSpringDashpotPressure
};

template<int dim>
struct BoundaryDescriptor
{
  // Dirichlet
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // Initial acceleration prescribed on Dirichlet boundary:
  // This data structure will only be used if the initial_acceleration is not set in FieldFunctions.
  // Moreover, this data structure will only be used for unsteady problems.
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    dirichlet_bc_initial_acceleration;

  // ComponentMask
  // If a certain boundary ID is not inserted into this map, it is assumed that all components are
  // active, in analogy to the default constructor of dealii::ComponentMask.
  std::map<dealii::types::boundary_id, dealii::ComponentMask> dirichlet_bc_component_mask;

  // Another type of Dirichlet boundary condition where the Dirichlet values come
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for the ALE mesh deformation problem in fluid-structure interaction).
  // ComponentMask is not implemented/available for this type of boundary condition.
  std::set<dealii::types::boundary_id> dirichlet_cached_bc;

  // Neumann
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // Robin boundary condition of the form
  // + ( v_h, k * d_h + c * d/dt(d_h) +  p * N )
  // or
  // + ( v_h, k * N * (d_h . N) + c * N . (d/dt(d_h) . N) + p * N )
  // using normal projections of displacement/velocity terms controlled via the
  // std::array<bool, 2> for the displacement (index 0) and velocity terms (index 1)
  // The std::array<double, 3> contains the parameters k (index 0), c (index 1) and p (index 2).
  mutable std::map<dealii::types::boundary_id,
                   std::pair<std::array<bool, 2>, std::array<double, 3>>>
    robin_k_c_p_param;

  // another type of Neumann boundary condition where the traction force comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Neumann boundary condition
  // is required for fluid-structure interaction problems)
  std::set<dealii::types::boundary_id> neumann_cached_bc;

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->dirichlet_cached_bc.find(boundary_id) != this->dirichlet_cached_bc.end())
      return BoundaryType::DirichletCached;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;
    else if(this->neumann_cached_bc.find(boundary_id) != this->neumann_cached_bc.end())
      return BoundaryType::NeumannCached;
    else if(this->robin_k_c_p_param.find(boundary_id) != this->robin_k_c_p_param.end() and
            this->neumann_cached_bc.find(boundary_id) == this->neumann_cached_bc.end())
    {
      // In FSI, the  interface is a BoundaryType::NeumannCached, where we also evaluate a Robin
      // term, but BoundaryType::RobinSpringDashpotPressure refers to spring/dashpot support, which
      // does not include the interface traction term in the FSI case.
      return BoundaryType::RobinSpringDashpotPressure;
    }

    AssertThrow(false,
                dealii::ExcMessage(
                  "Could not find a boundary type to the specified boundary_id = " +
                  std::to_string(boundary_id) +
                  ". A possible reason is that you "
                  "forgot to define a boundary condition for this boundary_id, or "
                  "that the boundary type associated to this boundary has not been "
                  "implemented."));

    return BoundaryType::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(
      dealii::types::boundary_id const             boundary_id,
      std::set<dealii::types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(dirichlet_bc.find(boundary_id) != dirichlet_bc.end())
    {
      counter++;

      AssertThrow(
        dirichlet_bc_component_mask.find(boundary_id) != dirichlet_bc_component_mask.end(),
        dealii::ExcMessage(
          "dirichlet_bc_component_mask must contain the same boundary IDs as dirichlet_bc."));

      AssertThrow(
        dirichlet_bc_initial_acceleration.find(boundary_id) !=
          dirichlet_bc_initial_acceleration.end(),
        dealii::ExcMessage(
          "dirichlet_bc_initial_acceleration must contain the same boundary IDs as dirichlet_bc."));
    }

    if(dirichlet_cached_bc.find(boundary_id) != dirichlet_cached_bc.end())
      counter++;

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(robin_k_c_p_param.find(boundary_id) != robin_k_c_p_param.end())
      counter++;

    if(neumann_cached_bc.find(boundary_id) != neumann_cached_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }

  void
  set_dirichlet_cached_data(
    std::shared_ptr<ContainerInterfaceData<1, dim, double> const> interface_data) const
  {
    dirichlet_cached_data = interface_data;
  }

  void
  set_neumann_cached_data(
    std::shared_ptr<ContainerInterfaceData<1, dim, double> const> interface_data) const
  {
    neumann_cached_data = interface_data;
  }

  std::shared_ptr<ContainerInterfaceData<1, dim, double> const>
  get_dirichlet_cached_data() const
  {
    AssertThrow(dirichlet_cached_data.get(),
                dealii::ExcMessage("Pointer to ContainerInterfaceData has not been initialized."));

    return dirichlet_cached_data;
  }

  std::shared_ptr<ContainerInterfaceData<1, dim, double> const>
  get_neumann_cached_data() const
  {
    AssertThrow(neumann_cached_data.get(),
                dealii::ExcMessage("Pointer to ContainerInterfaceData has not been initialized."));

    return neumann_cached_data;
  }

  std::map<dealii::types::boundary_id, std::pair<std::array<bool, 2>, std::array<double, 3>>>
  get_robin_k_c_p_param() const
  {
    return robin_k_c_p_param;
  }

  void
  set_robin_k_c_p_param(
    std::map<dealii::types::boundary_id, std::pair<std::array<bool, 2>, std::array<double, 3>>>
      robin_k_c_p_param_in) const
  {
    this->robin_k_c_p_param = robin_k_c_p_param_in;
  }

private:
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double> const> dirichlet_cached_data;
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double> const> neumann_cached_data;
};

} // namespace Structure
} // namespace ExaDG

#endif
