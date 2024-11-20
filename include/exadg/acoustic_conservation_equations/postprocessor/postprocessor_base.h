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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor_interface.h>
#include <exadg/acoustic_conservation_equations/spatial_discretization/spatial_operator.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim, typename Number>
class SpatialOperatorBase;

/*
 *  Base class for postprocessor of the acoustic conservation equations.
 */
template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  using VectorType = typename PostProcessorInterface<Number>::BlockVectorType;

  using AcousticsOperator = SpatialOperator<dim, Number>;

public:
  virtual ~PostProcessorBase() = default;
  /*
   * Setup function.
   */
  virtual void
  setup(AcousticsOperator const & pde_operator) = 0;
};


} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
