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

#ifndef INCLUDE_EXADG_OPERATORS_FINITE_ELEMENT_H_
#define INCLUDE_EXADG_OPERATORS_FINITE_ELEMENT_H_

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

  if(n_components == 1)
  {
    if(is_dg)
    {
      if(element_type == ElementType::Hypercube)
      {
        fe = std::make_shared<dealii::FE_DGQ<dim>>(degree);
      }
      else if(element_type == ElementType::Simplex)
      {
        fe = std::make_shared<dealii::FE_SimplexDGP<dim>>(degree);
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
        fe = std::make_shared<dealii::FE_Q<dim>>(degree);
      }
      else if(element_type == ElementType::Simplex)
      {
        fe = std::make_shared<dealii::FE_SimplexP<dim>>(degree);
      }
      else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    }
  }
  else
  {
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
        fe =
          std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexP<dim>(degree), n_components);
      }
      else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    }
  }

  return fe;
}

} // namespace ExaDG



#endif /* INCLUDE_EXADG_OPERATORS_FINITE_ELEMENT_H_ */
