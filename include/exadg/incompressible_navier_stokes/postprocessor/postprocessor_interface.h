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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<typename Number>
class PostProcessorInterface
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorInterface()
  {
  }

  /*
   * This function has to be called to apply the postprocessing tools.
   */
  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
