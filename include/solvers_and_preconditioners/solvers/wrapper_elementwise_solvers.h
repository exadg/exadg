/*
 * wrapper_elementwise_solvers.h
 *
 *  Created on: Oct 22, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
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
         int fe_degree,
         typename value_type,
         typename Operator,
         typename Preconditioner>
class IterativeSolver : public IterativeSolverBase<LinearAlgebra::distributed::Vector<value_type>>
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef IterativeSolver<dim, number_of_equations, fe_degree, value_type, Operator, Preconditioner>
    THIS;

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
  solve(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    op.get_data().cell_loop(&THIS::solve_elementwise, this, dst, src);

    return 0;
  }

private:
  void
  solve_elementwise(MatrixFree<dim, value_type> const &           data,
                    VectorType &                                  dst,
                    VectorType const &                            src,
                    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, number_of_equations, value_type> fe_eval(
      data, op.get_dof_index(), op.get_quad_index());

    const unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type>> solution(dofs_per_cell);

    // setup elementwise solver

    // CG
    std::shared_ptr<Elementwise::SolverCG<VectorizedArray<value_type>>> solver_cg;

    // GMRES
    std::shared_ptr<Elementwise::SolverGMRES<VectorizedArray<value_type>>> solver_gmres;

    if(iterative_solver_data.solver_type == SolverType::CG)
    {
      solver_cg.reset(
        new Elementwise::SolverCG<VectorizedArray<value_type>>(dofs_per_cell,
                                                               iterative_solver_data.solver_data));
    }
    else if(iterative_solver_data.solver_type == SolverType::GMRES)
    {
      solver_gmres.reset(new Elementwise::SolverGMRES<VectorizedArray<value_type>>(
        dofs_per_cell, iterative_solver_data.solver_data));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // loop over all cells and solve local problem iteratively on each cell
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src, 0);

      // initialize operator and preconditioner for current cell
      op.setup(cell);
      preconditioner.setup(cell);

      // call iterative solver and solve on current cell
      if(iterative_solver_data.solver_type == SolverType::CG)
      {
        solver_cg->solve(&op, solution.begin(), fe_eval.begin_dof_values(), &preconditioner);
      }
      else if(iterative_solver_data.solver_type == Elementwise::SolverType::GMRES)
      {
        solver_gmres->solve(&op, solution.begin(), fe_eval.begin_dof_values(), &preconditioner);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }

      // write solution on current element to global dof vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = solution[j];
      fe_eval.set_dof_values(dst, 0);
    }
  }

  Operator & op;

  Preconditioner & preconditioner;

  IterativeSolverData const iterative_solver_data;
};

} // namespace Elementwise

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_WRAPPER_ELEMENTWISE_SOLVERS_H_ \
        */
