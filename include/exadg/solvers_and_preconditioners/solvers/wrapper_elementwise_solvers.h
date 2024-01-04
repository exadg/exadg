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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/solvers_and_preconditioners/preconditioners/elementwise_preconditioners.h>
#include <exadg/solvers_and_preconditioners/solvers/elementwise_krylov_solvers.h>
#include <exadg/solvers_and_preconditioners/solvers/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>

namespace ExaDG
{
namespace Elementwise
{
/*
 * Solver data
 */
struct IterativeSolverData
{
  IterativeSolverData() : solver_type(Elementwise::Solver::CG), solver_data(SolverData())
  {
  }

  Solver solver_type;

  SolverData solver_data;
};

template<int dim,
         int number_of_equations,
         typename Number,
         typename Operator,
         typename Preconditioner>
class IterativeSolver
  : public Krylov::SolverBase<dealii::LinearAlgebra::distributed::Vector<Number>>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef IterativeSolver<dim, number_of_equations, Number, Operator, Preconditioner> THIS;

  IterativeSolver(Operator &                operator_in,
                  Preconditioner &          preconditioner_in,
                  IterativeSolverData const solver_data_in)
    : op(operator_in), preconditioner(preconditioner_in), iterative_solver_data(solver_data_in)
  {
  }

  virtual ~IterativeSolver()
  {
  }

  void
  update_preconditioner(bool const update_preconditioner) const override
  {
    if(preconditioner.needs_update() or update_preconditioner)
    {
      preconditioner.update();
    }
  }

  /**
   * Solve function. This function may be called with identical dst, src vectors.
   */
  unsigned int
  solve(VectorType & dst, VectorType const & src) const override
  {
    dst = 0;

    op.get_matrix_free().cell_loop(&THIS::solve_elementwise, this, dst, src);

    return 0;
  }

private:
  void
  solve_elementwise(dealii::MatrixFree<dim, Number> const &       matrix_free,
                    VectorType &                                  dst,
                    VectorType const &                            src,
                    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegrator<dim, number_of_equations, Number> integrator(matrix_free,
                                                                op.get_dof_index(),
                                                                op.get_quad_index());

    unsigned int const dofs_per_cell = integrator.dofs_per_cell;

    dealii::AlignedVector<dealii::VectorizedArray<Number>> solution(dofs_per_cell);

    // setup elementwise solver
    if(iterative_solver_data.solver_type == Solver::CG)
    {
      solver = std::make_shared<
        Elementwise::SolverCG<dealii::VectorizedArray<Number>, Operator, Preconditioner>>(
        dofs_per_cell, iterative_solver_data.solver_data);
    }
    else if(iterative_solver_data.solver_type == Solver::GMRES)
    {
      solver = std::make_shared<
        Elementwise::SolverGMRES<dealii::VectorizedArray<Number>, Operator, Preconditioner>>(
        dofs_per_cell, iterative_solver_data.solver_data);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    // loop over all cells and solve local problem iteratively on each cell
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src, 0);

      // initialize operator and preconditioner for current cell
      op.setup(cell, dofs_per_cell);
      preconditioner.setup(cell);

      // call iterative solver and solve on current cell
      solver->solve(&op, solution.begin(), integrator.begin_dof_values(), &preconditioner);

      // write solution on current element to global dof vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = solution[j];
      integrator.set_dof_values(dst, 0);
    }
  }

  mutable std::shared_ptr<
    Elementwise::SolverBase<dealii::VectorizedArray<Number>, Operator, Preconditioner>>
    solver;

  Operator & op;

  Preconditioner & preconditioner;

  IterativeSolverData const iterative_solver_data;
};

} // namespace Elementwise
} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_ \
        */
