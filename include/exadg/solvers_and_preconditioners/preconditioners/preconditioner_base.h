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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
template<typename value_type>
class PreconditionerBase
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<value_type> VectorType;

  PreconditionerBase() : update_needed(true)
  {
  }

  virtual ~PreconditionerBase()
  {
  }

  void
  set_update_flag()
  {
    update_needed = true;
  }

  bool
  needs_update() const
  {
    return update_needed;
  }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  update() = 0;

  virtual std::shared_ptr<TimerTree>
  get_timings() const
  {
    return std::make_shared<TimerTree>();
  }

protected:
  bool update_needed;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_ */
