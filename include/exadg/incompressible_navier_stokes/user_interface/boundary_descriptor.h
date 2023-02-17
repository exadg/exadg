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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// C/C++
#include <set>

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
//clang-format off
/*
 *
 *   Boundary conditions:
 *
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     example          |          velocity         |               pressure                         |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     inflow, no-slip  |   Dirichlet(Cached):      |  Neumann:                                      |
 *   |                      | prescribe g_u             | no BCs to be prescribed                        |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     slip             |   Slip:                   |  Neumann:                                      |
 *   |                      | no BCs to be prescribed   | no BCs to be prescribed                        |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     symmetry         |   Symmetry:               |  Neumann:                                      |
 *   |                      | no BCs to be prescribed   | no BCs to be prescribed                        |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     outflow          |   Neumann:                |  Dirichlet:                                    |
 *   |                      | prescribe F(u)*n          | prescribe g_p                                  |
 *   +----------------------+---------------------------+------------------------------------------------+
 *
 *   Divergence formulation: F(u) = F_nu(u) / nu = ( grad(u) + grad(u)^T )
 *   Laplace formulation:    F(u) = F_nu(u) / nu = grad(u)
 */
//clang-format on

enum class BoundaryTypeU
{
  Undefined,
  Dirichlet,
  DirichletCached,
  Slip,
  Neumann,
  Symmetry
};

enum class BoundaryTypeP
{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
struct BoundaryDescriptorU
{
  // Dirichlet: prescribe all components of the velocity
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // another type of Dirichlet boundary condition where the Dirichlet value comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for fluid-structure interaction problems)
  std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>> dirichlet_cached_bc;

  // Slip: the velocity normal to boundary is set to zero
  //
  //   u*n=0
  //
  // Only the boundary IDs are stored but no functions need to be evaluated.
  std::set<dealii::types::boundary_id> slip_bc;

  // Neumann: prescribe all components of the velocity gradient in normal direction
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // Symmetry: A boundary condition prescribing "symmetry" for planar surfaces. Using this boundary
  // condition, the velocity normal to boundary is set to zero
  //
  //   u*n=0 (1)
  //
  // as well as the derivative of tangential velocity components (u - (u*n) * n) in the direction
  // normal to the symmetry boundary. Mathematically, this can be written as
  //
  //   grad(u - (u*n) * n) * n = 0
  //
  // or (after algebraic manipulations)
  //
  //   grad(u)*n - [(grad(u)*n)*n] n = 0 . (2a)
  //
  // One can show that the same relation also holds for grad(u)^T if (u*n)=0 and if the normal
  // vector is constant, i.e. for a planar surface
  //
  //   grad(u)^T*n // n is constant
  //     = grad(u*n) // write gradient as sum of directional derivatives along n, t1, t2
  //     = (d(u*n)/dx_n) n + (d(u*n)/dx_t1) t1 + (d(u*n)/dx_t2) t2
  //     = (d(u*n)/dx_n) n
  //     = [grad(u*n)*n] n
  //     = [(grad(u)^T*n)*n] n
  //
  // where the derivative of the normal velocity in tangential directions t1, t2 has to vanish since
  // u*n=0 on the whole symmetry boundary. Hence, we obtain
  //
  //   grad(u)^T*n - [(grad(u)^T*n)*n] n = 0 . (2b)
  //
  // This implies that the surface stress vector F(u)*n in tangential directions is set to zero in
  // case of a symmetry boundary
  //
  //   F(u)*n - [(F(u)*n)*n] n = 0 .
  //
  // where F(u) = grad(u) (Laplace formulation) or F(u) = grad(u) + grad(u)^T (divergence
  // formulation). For the Laplace formulation, this equation follows immediately from equation
  // (2a). For the divergence formulation, this equation is obtained by simply adding equations (2a)
  // and (2b).
  //
  // Note that the user does not have to prescribe data for this symmetry boundary condition;
  // so one can simply use ZeroFunction<dim> (the function is not relevant because it will not
  // be evaluated by the code).
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> symmetry_bc;

  // add more types of boundary conditions


  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeU
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryTypeU::Dirichlet;
    else if(this->dirichlet_cached_bc.find(boundary_id) != this->dirichlet_cached_bc.end())
      return BoundaryTypeU::DirichletCached;
    else if(this->slip_bc.find(boundary_id) != this->slip_bc.end())
      return BoundaryTypeU::Slip;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryTypeU::Neumann;
    else if(this->symmetry_bc.find(boundary_id) != this->symmetry_bc.end())
      return BoundaryTypeU::Symmetry;

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
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      counter++;

    if(this->dirichlet_cached_bc.find(boundary_id) != this->dirichlet_cached_bc.end())
      counter++;

    if(this->slip_bc.find(boundary_id) != this->slip_bc.end())
      counter++;

    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      counter++;

    if(this->symmetry_bc.find(boundary_id) != this->symmetry_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

template<int dim>
struct BoundaryDescriptorP
{
  // Dirichlet: prescribe pressure value
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // Neumann: only the boundary IDs are stored but no inhomogeneous boundary conditions are
  // prescribed
  std::set<dealii::types::boundary_id> neumann_bc;

  // add more types of boundary conditions


  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeP
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryTypeP::Dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryTypeP::Neumann;

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
    velocity = std::make_shared<BoundaryDescriptorU<dim>>();
    pressure = std::make_shared<BoundaryDescriptorP<dim>>();
  }

  std::shared_ptr<BoundaryDescriptorU<dim>> velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> pressure;
};

template<int dim, typename Number>
inline void
verify_boundary_conditions(BoundaryDescriptor<dim> const & boundary_descriptor,
                           Grid<dim> const &               grid)
{
  ExaDG::verify_boundary_conditions(*boundary_descriptor.velocity, grid);
  ExaDG::verify_boundary_conditions(*boundary_descriptor.pressure, grid);
}

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
