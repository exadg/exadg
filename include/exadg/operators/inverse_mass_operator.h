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
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             InverseMassOperatorData const           inverse_mass_operator_data)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = inverse_mass_operator_data.dof_index;
    quad_index        = inverse_mass_operator_data.quad_index;

    data = inverse_mass_operator_data.parameters;

    dealii::FiniteElement<dim> const & fe = matrix_free->get_dof_handler(dof_index).get_fe();

    // The inverse mass operator is only available for discontinuous Galerkin discretizations
    AssertThrow(fe.conforms(dealii::FiniteElementData<dim>::L2),
                dealii::ExcMessage("InverseMassOperator only implemented for DG!"));

    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      // Currently, the inverse mass realized as matrix-free operator evaluation is only available
      // in deal.II for tensor-product elements.
      AssertThrow(
        fe.base_element(0).dofs_per_cell == dealii::Utilities::pow(fe.degree + 1, dim),
        dealii::ExcMessage(
          "The matrix-free inverse mass operator is currently only available for tensor-product elements."));

      // Currently, the inverse mass realized as matrix-free operator evaluation is only available
      // in deal.II if n_q_points_1d = n_nodes_1d.
      AssertThrow(
        this->matrix_free->get_shape_info(0, quad_index).data[0].n_q_points_1d == fe.degree + 1,
        dealii::ExcMessage(
          "The matrix-free inverse mass operator is currently only available if n_q_points_1d = n_nodes_1d."));
    }
    // We create a block-Jacobi preconditioner with MassOperator as underlying operator in case the
    // inverse mass can not be realized as a matrix-free operator.
    else if(data.implementation_type == InverseMassType::ElementwiseKrylovSolver or
            data.implementation_type == InverseMassType::BlockMatrices)
    {
      // initialize mass operator
      dealii::AffineConstraints<Number> constraint;
      constraint.clear();
      constraint.close();

      MassOperatorData<dim> mass_operator_data;
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

      mass_operator.initialize(*matrix_free, constraint, mass_operator_data);

      block_jacobi_preconditioner =
        std::make_shared<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
          mass_operator, true /* initialize_preconditioner */);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified InverseMassType is not implemented."));
    }
  }

  /**
   * Updates the inverse mass operator. This function recomputes the diagonal/block-diagonal in case
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
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified InverseMassType is not implemented."));
    }
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst.zero_out_ghost_values();

    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      matrix_free->cell_loop(&This::cell_loop_matrix_free_operator, this, dst, src);
    }
    else // ElementwiseKrylovSolver or BlockMatrices
    {
      block_jacobi_preconditioner->vmult(dst, src);
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

  // This variable is only relevant if the inverse mass can not be realized as a matrix-free
  // operator. Since this class allows only L2-conforming spaces (discontinuous Galerkin method),
  // the mass matrix is block-diagonal and a block-Jacobi preconditioner inverts the mass operator
  // exactly (up to solver tolerances). The implementation of the block-Jacobi preconditioner can be
  // matrix-based or matrix-free, depending on the parameters specified.
  std::shared_ptr<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>
    block_jacobi_preconditioner;

  // In case we realize the inverse mass as block-Jacobi preconditioner, we need a MassOperator as
  // underlying operator for the block-Jacobi preconditioner.
  MassOperator<dim, n_components, Number> mass_operator;
};

struct InverseMassOperatorDataHdiv
{
  InverseMassOperatorDataHdiv() : dof_index(0), quad_index(0)
  {
  }

  // Parameters referring to dealii::MatrixFree
  unsigned int dof_index;
  unsigned int quad_index;

  InverseMassParametersHdiv parameters;
};

/*
 * Inverse mass operator for H(div)-conforming space:
 *
 * This class applies the inverse mass operator by solving the mass system as a global linear system
 * of equations for all degrees of freedom. It is used in case the mass operator is not
 * block-diagonal and can not be inverted element-wise (e.g. H(div)-conforming space).
 */
template<int dim, int n_components, typename Number>
class InverseMassOperatorHdiv
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & constraints,
             InverseMassOperatorDataHdiv const         inverse_mass_operator_data)
  {
    // mass operator
    MassOperatorData<dim> mass_operator_data;
    mass_operator_data.dof_index  = inverse_mass_operator_data.dof_index;
    mass_operator_data.quad_index = inverse_mass_operator_data.quad_index;
    mass_operator.initialize(matrix_free, constraints, mass_operator_data);

    Krylov::SolverDataCG solver_data;
    solver_data.max_iter             = inverse_mass_operator_data.parameters.solver_data.max_iter;
    solver_data.solver_tolerance_abs = inverse_mass_operator_data.parameters.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = inverse_mass_operator_data.parameters.solver_data.rel_tol;

    if(inverse_mass_operator_data.parameters.preconditioner == PreconditionerMass::None)
    {
      solver_data.use_preconditioner = false;
    }
    else if(inverse_mass_operator_data.parameters.preconditioner == PreconditionerMass::PointJacobi)
    {
      preconditioner =
        std::make_shared<JacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
          mass_operator, true /* initialize_preconditioner */);

      solver_data.use_preconditioner = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    solver =
      std::make_shared<Krylov::SolverCG<MassOperator<dim, n_components, Number>,
                                        PreconditionerBase<Number>,
                                        VectorType>>(mass_operator, *preconditioner, solver_data);
  }

  /**
   * This function applies the inverse mass operator. Note that this function allows identical dst,
   * src vector, i.e. the function can be called like apply(dst, dst).
   */
  unsigned int
  apply(VectorType & dst, VectorType const & src) const
  {
    Assert(solver.get() != 0, dealii::ExcMessage("Mass solver has not been initialized."));

    VectorType temp;

    // Note that the inverse mass operator might be called like inverse_mass.apply(dst, dst),
    // i.e. with identical destination and source vectors. In this case, we need to make sure
    // that the result is still correct.
    if(&dst == &src)
    {
      temp = src;
      return solver->solve(dst, temp);
    }
    else
    {
      return solver->solve(dst, src);
    }
  }

private:
  // Solver/preconditioner for mass system solving a global linear system of equations for all
  // degrees of freedom.
  std::shared_ptr<PreconditionerBase<Number>>     preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> solver;

  // We need a MassOperator as underlying operator.
  MassOperator<dim, n_components, Number> mass_operator;

  InverseMassParametersHdiv data;
};

} // namespace ExaDG


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
