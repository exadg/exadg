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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_FOR_RANS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_FOR_RANS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include <exadg/incompressible_navier_stokes_for_rans/postprocessor/postprocessor_interface.h>

namespace ExaDG
{
namespace IncRANS
{
template<int dim, typename Number>
class SpatialOperatorBase;

/*
 *  Base class for postprocessor of the incompressible Navier-Stokes equation.
 */
template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

  typedef SpatialOperatorBase<dim, Number> Operator;

public:
  virtual ~PostProcessorBase()
  {
  }

  /*
   * Setup function.
   */
  virtual void
  setup(Operator const & pde_operator) = 0;
};


} // namespace IncRANS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_FOR_RANS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ \
        */
