/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2024 by the ExaDG authors
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

#ifndef EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERPOLATE_H_
#define EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERPOLATE_H_

#include <deal.II/numerics/vector_tools.h>

#include <type_traits>

namespace ExaDG
{
namespace Utilities
{
template<int dim,
         int spacedim,
         typename Number,
         template<typename>
         typename VectorType,
         typename FunctionNumber>
void
interpolate(dealii::DoFHandler<dim, spacedim> const &          dof_handler,
            dealii::Function<spacedim, FunctionNumber> const & function,
            VectorType<Number> &                               vec)
{
  if constexpr(std::is_same_v<Number, FunctionNumber>)
  {
    dealii::VectorTools::interpolate(dof_handler, function, vec);
  }
  else
  {
    VectorType<FunctionNumber> vec_fn;
    vec_fn = vec;
    dealii::VectorTools::interpolate(dof_handler, function, vec_fn);
    vec = vec_fn;
  }
}

template<int dim, int spacedim, typename VectorType, typename FunctionNumber>
void
interpolate(dealii::DoFHandler<dim, spacedim> const &    dof_handler,
            dealii::Function<spacedim, FunctionNumber> & function,
            VectorType &                                 vec,
            double const                                 time)
{
  function.set_time(time);
  interpolate(dof_handler, function, vec);
}

template<int dim,
         int spacedim,
         typename Number,
         template<typename>
         typename VectorType,
         typename FunctionNumber>
void
interpolate(dealii::Mapping<dim, spacedim> const &             mapping,
            dealii::DoFHandler<dim, spacedim> const &          dof_handler,
            dealii::Function<spacedim, FunctionNumber> const & function,
            VectorType<Number> &                               vec)
{
  if constexpr(std::is_same_v<Number, FunctionNumber>)
  {
    dealii::VectorTools::interpolate(mapping, dof_handler, function, vec);
  }
  else
  {
    VectorType<FunctionNumber> vec_fn;
    vec_fn = vec;
    dealii::VectorTools::interpolate(mapping, dof_handler, function, vec_fn);
    vec = vec_fn;
  }
}

template<int dim, int spacedim, typename VectorType, typename FunctionNumber>
void
interpolate(dealii::Mapping<dim, spacedim> const &       mapping,
            dealii::DoFHandler<dim, spacedim> const &    dof_handler,
            dealii::Function<spacedim, FunctionNumber> & function,
            VectorType &                                 vec,
            double const                                 time)
{
  function.set_time(time);
  interpolate(mapping, dof_handler, function, vec);
}


} // namespace Utilities
} // namespace ExaDG


#endif /*EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERPOLATE_H_*/
