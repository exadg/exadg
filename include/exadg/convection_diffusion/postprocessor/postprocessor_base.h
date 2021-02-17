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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/analytical_solution.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim, typename Number>
class Operator;

template<typename Number>
class PostProcessorInterface
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~PostProcessorInterface()
  {
  }

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  virtual void
  setup(Operator<dim, Number> const & pde_operator, Mapping<dim> const & mapping) = 0;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
