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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

#include <deal.II/lac/la_parallel_block_vector.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace Acoustics
{
template<typename Number>
class PostProcessorInterface
{
protected:
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

public:
  virtual ~PostProcessorInterface() = default;

  /*
   * This function has to be called to apply the postprocessing tools.
   */
  virtual void
  do_postprocessing(BlockVectorType const & solution,
                    double const            time             = 0.0,
                    types::time_step const  time_step_number = numbers::steady_timestep) = 0;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
