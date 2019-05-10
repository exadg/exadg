/*
 * wrapper_elementwise_solvers.h
 *
 *  Created on: Oct 22, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/operators.h>

#include "../preconditioner/elementwise_preconditioners.h"
#include "elementwise_krylov_solvers.h"
#include "iterative_solvers_dealii_wrapper.h"

using namespace dealii;

namespace Elementwise
{
/*
 * Solver data
 */
struct IterativeSolverData
{
  IterativeSolverData() : solver_type(Elementwise::SolverType::CG), solver_data(SolverData())
  {
  }

  SolverType solver_type;

  SolverData solver_data;
};

template<int dim,
         int number_of_equations,
         typename Number,
         typename Operator,
         typename Preconditioner>
class IterativeSolver : public IterativeSolverBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

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

  unsigned int
  solve(VectorType & dst, VectorType const & src, bool const /* update_preconditioner */) const
  {
    dst = 0;

    op.get_data().cell_loop(&THIS::solve_elementwise, this, dst, src);

    return 0;
  }

private:
  void
  solve_elementwise(MatrixFree<dim, Number> const &               matrix_free,
                    VectorType &                                  dst,
                    VectorType const &                            src,
                    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegrator<dim, number_of_equations, Number> integrator(matrix_free,
                                                                op.get_dof_index(),
                                                                op.get_quad_index());

    const unsigned int dofs_per_cell = integrator.dofs_per_cell;

    AlignedVector<VectorizedArray<Number>> solution(dofs_per_cell);

    // setup elementwise solver
    if(iterative_solver_data.solver_type == SolverType::CG)
    {
      solver.reset(new Elementwise::SolverCG<VectorizedArray<Number>, Operator, Preconditioner>(
        dofs_per_cell, iterative_solver_data.solver_data));
    }
    else if(iterative_solver_data.solver_type == SolverType::GMRES)
    {
      solver.reset(new Elementwise::SolverGMRES<VectorizedArray<Number>, Operator, Preconditioner>(
        dofs_per_cell, iterative_solver_data.solver_data));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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
    Elementwise::SolverBase<VectorizedArray<Number>, Operator, Preconditioner>>
    solver;

  Operator & op;

  Preconditioner & preconditioner;

  IterativeSolverData const iterative_solver_data;
};

} // namespace Elementwise

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_ \
        */
