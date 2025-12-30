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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */
#ifndef EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace Structure
{
// forward declaration
template<int dim, typename Number>
class Operator;

/*
 * Base class for postprocessor of displacement-based (hyper-)elasticity.
 */
template<int dim, typename Number>
class PostProcessorBase
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  /*
   * Setup function.
   */
  virtual void
  setup(Operator<dim, Number> const & pde_operator) = 0;

  /*
   * Postprocessing function.
   */
  virtual void
  do_postprocessing(VectorType const &     solution,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) = 0;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
