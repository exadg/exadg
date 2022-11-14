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
enum class SolutionFieldType
{
  scalar,
  vector,
  cellwise
};

template<int dim, typename Number>
class SolutionField : public dealii::Subscriptor
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  SolutionField()
    : recompute_solution_field([](VectorType &, VectorType const &, bool const) {}),
      type(SolutionFieldType::scalar),
      name("solution"),
      dof_handler(nullptr),
      is_available(false),
      solution(nullptr)
  {
  }

  void
  reinit(VectorType const & solution_in)
  {
    is_available = false;
    solution     = &solution_in;
  }

  void
  evaluate(bool const unsteady)
  {
    if(!is_available)
    {
      recompute_solution_field(vector, *solution, unsteady);
      is_available = true;
    }
  }

  VectorType &
  get_vector_reference()
  {
    return vector;
  }

  VectorType const &
  get_vector() const
  {
    AssertThrow(is_available,
                dealii::ExcMessage("You are trying to access a Vector that is not available."));

    return vector;
  }

  std::string const &
  get_name() const
  {
    return name;
  }

  dealii::DoFHandler<dim> const &
  get_dof_handler() const
  {
    return *dof_handler;
  }

  SolutionFieldType
  get_type() const
  {
    return type;
  }

  std::function<void(VectorType &, VectorType const &, bool const)> recompute_solution_field;

  SolutionFieldType type;

  std::string name;

  dealii::DoFHandler<dim> const * dof_handler;

private:
  bool               is_available;
  VectorType         vector;
  VectorType const * solution;
};

template<int dim, typename Number>
std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const &
evaluate_get(std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> & vec,
             const bool                                                      unsteady)
{
  for(auto & v : vec)
    v->evaluate(unsteady);
  return vec;
}

template<int dim, typename Number>
typename SolutionField<dim, Number>::VectorType const &
evaluate_get(SolutionField<dim, Number> & sol, const bool unsteady)
{
  sol.evaluate(unsteady);
  return sol.get_vector();
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_SOLUTION_FIELD_H_ */
