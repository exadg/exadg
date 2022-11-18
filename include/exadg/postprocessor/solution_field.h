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
    : initialize_vector([](std::shared_ptr<VectorType> &) {}),
      recompute_solution_field([](VectorType &, VectorType const &) {}),
      type(SolutionFieldType::scalar),
      name("solution"),
      dof_handler(nullptr),
      is_available(false)
  {
  }

  /**
   * This function initializes the DoF vector, the main data object of this class.
   * This is done by using the lambda initialize_vector. The vector solution_vector
   * may also point to external data.
   */
  void
  reinit()
  {
    initialize_vector(solution_vector);
  }

  /**
   * This function invalidates the solution vector.
   */
  void
  invalidate()
  {
    is_available = false;
  }

  void
  evaluate(VectorType const & src)
  {
    if(!is_available)
    {
      recompute_solution_field(*solution_vector, src);
      is_available = true;
    }
  }

  VectorType const &
  get() const
  {
    AssertThrow(is_available,
                dealii::ExcMessage("You are trying to access a Vector that is invalid."));

    return *solution_vector;
  }

  VectorType const &
  evaluate_get(VectorType const & src)
  {
    evaluate(src);

    return get();
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

  // TODO: these element variables should not be public but instead be passed to the reinit function
  std::function<void(std::shared_ptr<VectorType> &)> initialize_vector;

  std::function<void(VectorType &, VectorType const &)> recompute_solution_field;

  SolutionFieldType type;

  std::string name;

  dealii::DoFHandler<dim> const * dof_handler;

private:
  mutable bool                        is_available;
  mutable std::shared_ptr<VectorType> solution_vector;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_SOLUTION_FIELD_H_ */
