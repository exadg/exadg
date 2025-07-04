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

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/inverse_mass_parameters.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
struct InverseMassOperatorData
{
  InverseMassOperatorData() : dof_index(0), quad_index(0)
  {
  }

  // Parameters referring to dealii::MatrixFree
  unsigned int dof_index;
  unsigned int quad_index;

  InverseMassParameters parameters;
};

template<int dim, int n_components, typename Number>
class InverseMassOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    InverseMassAsMatrixFreeOperator;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  InverseMassOperator() : matrix_free(nullptr), dof_index(0), quad_index(0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free_in,
             InverseMassOperatorData const             inverse_mass_operator_data,
             dealii::AffineConstraints<Number> const * constraints = nullptr)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = inverse_mass_operator_data.dof_index;
    quad_index        = inverse_mass_operator_data.quad_index;

    data = inverse_mass_operator_data.parameters;

    dealii::FiniteElement<dim> const & fe = matrix_free->get_dof_handler(dof_index).get_fe();

    // Some implementation variants of the inverse mass operator are based on assumptions on the
    // discretization and more efficient choices are available for tensor-product elements or
    // L2-conforming spaces.
    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      AssertThrow(
        fe.base_element(0).dofs_per_cell == dealii::Utilities::pow(fe.degree + 1, dim),
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator is currently only available for isotropic tensor-product elements."));

      AssertThrow(
        this->matrix_free->get_shape_info(0, quad_index).data[0].n_q_points_1d == fe.degree + 1,
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator is currently only available if n_q_points_1d = n_nodes_1d."));

      AssertThrow(
        fe.conforms(dealii::FiniteElementData<dim>::L2),
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator is only available for L2-conforming elements."));
    }
    else
    {
      // Setup MassOperator as underlying operator for cell-wise direct/iterative inverse or global
      // solve.
      MassOperatorData<dim, Number> mass_operator_data;
      mass_operator_data.dof_index  = dof_index;
      mass_operator_data.quad_index = quad_index;

      if(data.implementation_type == InverseMassType::ElementwiseKrylovSolver)
      {
        mass_operator_data.implement_block_diagonal_preconditioner_matrix_free = true;
        mass_operator_data.solver_block_diagonal = Elementwise::Solver::CG;
        if(inverse_mass_operator_data.parameters.preconditioner == PreconditionerMass::None)
        {
          mass_operator_data.preconditioner_block_diagonal = Elementwise::Preconditioner::None;
        }
        else if(inverse_mass_operator_data.parameters.preconditioner ==
                PreconditionerMass::PointJacobi)
        {
          mass_operator_data.preconditioner_block_diagonal =
            Elementwise::Preconditioner::PointJacobi;
        }
        mass_operator_data.solver_data_block_diagonal =
          inverse_mass_operator_data.parameters.solver_data;
      }

      // Use constraints if provided.
      if(constraints == nullptr)
      {
        dealii::AffineConstraints<Number> dummy_constraints;
        dummy_constraints.close();
        mass_operator.initialize(*matrix_free, dummy_constraints, mass_operator_data);
      }
      else
      {
        mass_operator.initialize(*matrix_free, *constraints, mass_operator_data);
      }

      if(data.implementation_type == InverseMassType::ElementwiseKrylovSolver or
         data.implementation_type == InverseMassType::BlockMatrices)
      {
        // Non-L2-conforming elements are asserted here because the cell-wise inverse neglecting
        // cell-coupling terms is only an approximation of the inverse mass matrix.
        AssertThrow(
          fe.conforms(dealii::FiniteElementData<dim>::L2),
          dealii::ExcMessage(
            "The cell-wise inverse mass operator is only available for L2-conforming elements."));

        block_jacobi_preconditioner =
          std::make_shared<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
            mass_operator, true /* initialize_preconditioner */);
      }
      else if(data.implementation_type == InverseMassType::GlobalKrylovSolver)
      {
        Krylov::SolverDataCG solver_data;
        solver_data.max_iter = inverse_mass_operator_data.parameters.solver_data.max_iter;
        solver_data.solver_tolerance_abs =
          inverse_mass_operator_data.parameters.solver_data.abs_tol;
        solver_data.solver_tolerance_rel =
          inverse_mass_operator_data.parameters.solver_data.rel_tol;

        solver_data.use_preconditioner =
          inverse_mass_operator_data.parameters.preconditioner != PreconditionerMass::None;
        if(inverse_mass_operator_data.parameters.preconditioner == PreconditionerMass::None)
        {
          // no setup required.
        }
        else if(inverse_mass_operator_data.parameters.preconditioner ==
                PreconditionerMass::PointJacobi)
        {
          global_preconditioner =
            std::make_shared<JacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
              mass_operator, true /* initialize_preconditioner */);
        }
        else if(inverse_mass_operator_data.parameters.preconditioner ==
                PreconditionerMass::BlockJacobi)
        {
          global_preconditioner =
            std::make_shared<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
              mass_operator, true /* initialize_preconditioner */);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("This `PreconditionerMass` is not implemented."));
        }

        global_solver = std::make_shared<Krylov::SolverCG<MassOperator<dim, n_components, Number>,
                                                          PreconditionerBase<Number>,
                                                          VectorType>>(mass_operator,
                                                                       *global_preconditioner,
                                                                       solver_data);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("The specified `InverseMassType` is not implemented."));
      }
    }
  }

  /**
   * Updates the inverse mass operator. This function recomputes the preconditioners in case
   * the geometry has changed (e.g. the mesh has been deformed).
   */
  void
  update()
  {
    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      // no updates needed as long as the MatrixFree object is up-to-date (which is not the
      // responsibility of the present class).
    }
    else if(data.implementation_type == InverseMassType::ElementwiseKrylovSolver or
            data.implementation_type == InverseMassType::BlockMatrices)
    {
      // the mass operator does not need to be updated as long as the MatrixFree object is
      // up-to-date (which is not the responsibility of the present class).

      // update the matrix-based components of the block-Jacobi preconditioner
      block_jacobi_preconditioner->update();
    }
    else if(data.implementation_type == InverseMassType::GlobalKrylovSolver)
    {
      global_preconditioner->update();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified InverseMassType is not implemented."));
    }
  }

  // dst = M^-1 * src
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    if(data.implementation_type == InverseMassType::GlobalKrylovSolver)
    {
      AssertThrow(global_solver.get() != 0,
                  dealii::ExcMessage("Global mass solver has not been initialized."));

      // Note that the inverse mass operator might be called like inverse_mass.apply(dst, dst),
      // i.e. with identical destination and source vectors. In this case, we need to make sure
      // that the result is still correct.
      if(&dst == &src)
      {
        VectorType tmp = src;
        global_solver->solve(dst, tmp);
      }
      else
      {
        global_solver->solve(dst, src);
      }
    }
    else
    {
      dst.zero_out_ghost_values();

      if(data.implementation_type == InverseMassType::MatrixfreeOperator)
      {
        matrix_free->cell_loop(&This::cell_loop_matrix_free_operator, this, dst, src);
      }
      else // ElementwiseKrylovSolver or BlockMatrices
      {
        AssertThrow(block_jacobi_preconditioner.get() != 0,
                    dealii::ExcMessage(
                      "Cell-wise iterative/direct block-Jacobi solver has not been initialized."));
        block_jacobi_preconditioner->vmult(dst, src);
      }
    }
  }

  // dst = scaling_factor * (M^-1 * src)
  void
  apply_scale(VectorType & dst, double const scaling_factor, VectorType const & src) const
  {
    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      // In the InverseMassType::MatrixfreeOperator case we can avoid
      // streaming the vector from memory twice.

      // ghost have to be zeroed out before MatrixFree::cell_loop().
      dst.zero_out_ghost_values();

      matrix_free->cell_loop(
        &This::cell_loop_matrix_free_operator,
        this,
        dst,
        src,
        /*operation before cell operation*/ {}, /*operation after cell operation*/
        [&](const unsigned int start_range, const unsigned int end_range) {
          for(unsigned int i = start_range; i < end_range; ++i)
            dst.local_element(i) *= scaling_factor;
        },
        dof_index);
    }
    else
    {
      apply(dst, src);
      dst *= scaling_factor;
    }
  }


private:
  void
  cell_loop_matrix_free_operator(dealii::MatrixFree<dim, Number> const &,
                                 VectorType &       dst,
                                 VectorType const & src,
                                 Range const &      cell_range) const
  {
    Integrator                      integrator(*matrix_free, dof_index, quad_index);
    InverseMassAsMatrixFreeOperator inverse_mass(integrator);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src, 0);

      inverse_mass.apply(integrator.begin_dof_values(), integrator.begin_dof_values());

      integrator.set_dof_values(dst, 0);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;

  InverseMassParameters data;

  // Solver and preconditioner for solving a global linear system of equations for all degrees of
  // freedom.
  std::shared_ptr<PreconditionerBase<Number>>     global_preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> global_solver;

  // Block-Jacobi preconditioner used as cell-wise inverse for L2-conforming spaces. In this case,
  // the mass matrix is block-diagonal and a block-Jacobi preconditioner inverts the mass operator
  // exactly (up to solver tolerances). The implementation of the block-Jacobi preconditioner can be
  // matrix-based or matrix-free, depending on the parameters specified.
  std::shared_ptr<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>
    block_jacobi_preconditioner;

  // MassOperator as underlying operator for the cell-wise or global iterative solves.
  MassOperator<dim, n_components, Number> mass_operator;
};
} // namespace ExaDG


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
