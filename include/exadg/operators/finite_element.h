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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_FINITE_ELEMENT_H_
#define EXADG_OPERATORS_FINITE_ELEMENT_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
/**
 * Creates and returns a shared pointer to dealii::FiniteElement<dim> depending on the element type,
 * the type of function space, the number of components of the solution field, and the polynomial
 * degree of the shape functions. Regarding the function space, the code assumes an H^1 conforming
 * space (Continuous Galerkin) if is_dg (= L^2-conforming space, Discontinuous Galerkin) is false.
 */
template<int dim>
std::shared_ptr<dealii::FiniteElement<dim>>
create_finite_element(ElementType const & element_type,
                      bool const          is_dg,
                      unsigned int const  n_components,
                      unsigned int const  degree)
{
  std::shared_ptr<dealii::FiniteElement<dim>> fe;

  // For n_components = 1, we would not need FESystem around the finite element. We do this
  // nevertheless in order to use the same code path for all cases, since this is not a
  // performance-critical part of the code.

  if(is_dg)
  {
    if(element_type == ElementType::Hypercube)
    {
      fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_DGQ<dim>(degree), n_components);
    }
    else if(element_type == ElementType::Simplex)
    {
      fe =
        std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexDGP<dim>(degree), n_components);
    }
    else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  }
  else
  {
    if(element_type == ElementType::Hypercube)
    {
      fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_Q<dim>(degree), n_components);
    }
    else if(element_type == ElementType::Simplex)
    {
      fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexP<dim>(degree), n_components);
    }
    else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  }

  return fe;
}

inline unsigned int
get_dofs_per_element_simplex_scalar(unsigned int const n_points_1d, unsigned int const dim)
{
  if(n_points_1d == 0)
    return 0;
  else if(n_points_1d == 1)
    return 1;
  else
  {
    unsigned int scalar_inner_dofs = 1;
    for(unsigned int d = 0; d < dim; ++d)
      scalar_inner_dofs = (scalar_inner_dofs * (n_points_1d + d)) / (d + 1);

    return scalar_inner_dofs;
  }
}

inline double
get_dofs_per_element(ExaDG::ElementType const element_type,
                     bool const               is_dg,
                     unsigned int const       n_components,
                     unsigned int const       degree,
                     unsigned int const       dim)
{
  double scalar_dofs_per_element = 0.;

  if(element_type == ElementType::Hypercube)
  {
    unsigned int n_points_1d = degree;
    if(is_dg)
      n_points_1d += 1;

    scalar_dofs_per_element = dealii::Utilities::pow(n_points_1d, dim);
  }
  else if(element_type == ElementType::Simplex)
  {
    unsigned int n_points_1d = degree + 1;

    if(is_dg)
    {
      scalar_dofs_per_element = get_dofs_per_element_simplex_scalar(n_points_1d, dim);
    }
    else // continuous Galerkin
    {
      if(dim == 2)
      {
        // consider a quadrilateral element that is subdivided into 2 triangular elements; the
        // quadrilateral element has one unique corner shared by 2 triangular elements; each
        // face/edge is shared by 2 triangular elements; inner dofs are unique per triangular
        // element

        scalar_dofs_per_element = 1. / 2.;                      // corners
        scalar_dofs_per_element += 3. * (n_points_1d - 2) / 2.; // faces/edges
        if(n_points_1d >= 3)
        {
          scalar_dofs_per_element +=
            get_dofs_per_element_simplex_scalar(n_points_1d - 3, dim); // inner
        }
      }
      else if(dim == 3)
      {
        // consider a hexahedral element that is subdivided into 5 tetrahedral elements; the
        // "hexahedral element" has one unique corner shared by 5 tetrahedral elements; assume that
        // each of the 6 unique edges of the "hexahedral element" (with 5 tets inside) is shared by
        // 5 tetrahedra; and each of the 4 unique faces per tetrahedral element is shared by
        // 2 tetrahedra; inner dofs are unique per tetrahedral element

        scalar_dofs_per_element = 1. / 5.;                      // corners
        scalar_dofs_per_element += 6. * (n_points_1d - 2) / 5.; // edges
        if(n_points_1d >= 3)
        {
          scalar_dofs_per_element +=
            4. * get_dofs_per_element_simplex_scalar(n_points_1d - 3, dim - 1) / 2.; // faces
        }
        if(n_points_1d >= 4)
        {
          scalar_dofs_per_element +=
            get_dofs_per_element_simplex_scalar(n_points_1d - 4, dim); // inner
        }
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  return scalar_dofs_per_element * n_components;
}

} // namespace ExaDG

#endif /* EXADG_OPERATORS_FINITE_ELEMENT_H_ */
