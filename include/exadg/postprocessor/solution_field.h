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

#ifndef INCLUDE_EXADG_POSTPROCESSOR_SOLUTION_FIELD_H_
#define INCLUDE_EXADG_POSTPROCESSOR_SOLUTION_FIELD_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
using namespace dealii;

enum class SolutionFieldType
{
  scalar,
  vector,
  cellwise
};

template<int dim, typename Number>
class SolutionField
{
public:
  SolutionField()
    : type(SolutionFieldType::scalar), name("solution"), dof_handler(nullptr), vector(nullptr)
  {
  }

  SolutionFieldType type;

  std::string name;

  DoFHandler<dim> const * dof_handler;

  LinearAlgebra::distributed::Vector<Number> const * vector;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_SOLUTION_FIELD_H_ */
