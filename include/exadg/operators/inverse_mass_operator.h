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

#ifndef EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_
#define EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_

// deal.II
#include <deal.II/fe/fe_data.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/inverse_mass_parameters.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
template<typename Number>
struct InverseMassOperatorData
{
  InverseMassOperatorData()
    : dof_index(0),
      quad_index(0),
      coefficient_is_variable(false),
      consider_inverse_coefficient(false),
      variable_coefficients(nullptr)
  {
  }

  // Get optimal in the sense of (most likely) fastest implementation type of the inverse mass
  // operator depending on the approximation space.
  template<int dim>
  static InverseMassType
  get_optimal_inverse_mass_type(dealii::FiniteElement<dim> const & fe,
                                ElementType const                  element_type)
  {
    if(fe.conforms(dealii::FiniteElementData<dim>::L2))
    {
      if(element_type == ElementType::Hypercube)
      {
        return InverseMassType::MatrixfreeOperator;
      }
      else
      {
        return InverseMassType::ElementwiseKrylovSolver;
      }
    }
    else
    {
      return InverseMassType::GlobalKrylovSolver;
    }
  }

  // Parameters referring to dealii::MatrixFree
  unsigned int dof_index;
  unsigned int quad_index;

  InverseMassParameters parameters;

  // Enable variable coefficients.
  bool coefficient_is_variable;

  // Consider the regular form of the coefficient (1) or its inverse (2):
  // (1) : (u_h , v_h * c)_Omega
  // (2) : (u_h , v_h / c)_Omega
  bool consider_inverse_coefficient;

  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;
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
             InverseMassOperatorData<Number> const     inverse_mass_operator_data,
             dealii::AffineConstraints<Number> const * constraints = nullptr)
  {
    this->matrix_free = &matrix_free_in;
    dof_index         = inverse_mass_operator_data.dof_index;
    quad_index        = inverse_mass_operator_data.quad_index;

    data = inverse_mass_operator_data.parameters;

    coefficient_is_variable      = inverse_mass_operator_data.coefficient_is_variable;
    consider_inverse_coefficient = inverse_mass_operator_data.consider_inverse_coefficient;
    variable_coefficients        = inverse_mass_operator_data.variable_coefficients;

    // Variable coefficients only implemented for the matrix-free operator.
    AssertThrow(not coefficient_is_variable or variable_coefficients != nullptr,
                dealii::ExcMessage("Pointer to variable coefficients not set properly."));

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
      mass_operator_data.dof_index                    = dof_index;
      mass_operator_data.quad_index                   = quad_index;
      mass_operator_data.coefficient_is_variable      = coefficient_is_variable;
      mass_operator_data.variable_coefficients        = variable_coefficients;
      mass_operator_data.consider_inverse_coefficient = consider_inverse_coefficient;

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
      global_solver->solve(dst, src);
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

  // dst = M^-1 * src
  void
  apply(VectorType &                                                        dst,
        VectorType const &                                                  src,
        const std::function<void(const unsigned int, const unsigned int)> & before_loop,
        const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
  {
    dst.zero_out_ghost_values();

    if(data.implementation_type == InverseMassType::MatrixfreeOperator)
    {
      matrix_free->cell_loop(
        &This::cell_loop_matrix_free_operator, this, dst, src, before_loop, after_loop);
    }
    else // ElementwiseKrylovSolver or BlockMatrices
    {
      if(before_loop)
        before_loop(0, src.locally_owned_size());
      block_jacobi_preconditioner->vmult(dst, src);
      if(after_loop)
        after_loop(0, src.locally_owned_size());
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

    if(coefficient_is_variable)
    {
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src, 0);

        dealii::AlignedVector<dealii::VectorizedArray<Number>> inverse_JxW_times_coefficient(
          integrator.n_q_points);
        inverse_mass.fill_inverse_JxW_values(inverse_JxW_times_coefficient);

        if(consider_inverse_coefficient)
        {
          // Consider a mass matrix of the form
          // (u_h , v_h / c)_Omega
          // hence fill the vector with (J / c)^-1 = c/J
          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            inverse_JxW_times_coefficient[q] *=
              this->variable_coefficients->get_coefficient_cell(cell, q);
          }
        }
        else
        {
          // Consider a mass matrix of the form
          // (u_h , v_h * c)_Omega
          // hence fill the vector with inv(J * c) = 1/(J * c)
          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            inverse_JxW_times_coefficient[q] /=
              this->variable_coefficients->get_coefficient_cell(cell, q);
          }
        }

        inverse_mass.apply(inverse_JxW_times_coefficient,
                           n_components,
                           integrator.begin_dof_values(),
                           integrator.begin_dof_values());

        integrator.set_dof_values(dst, 0);
      }
    }
    else
    {
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src, 0);

        inverse_mass.apply(integrator.begin_dof_values(), integrator.begin_dof_values());

        integrator.set_dof_values(dst, 0);
      }
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;

  InverseMassParameters data;

  // Variable coefficients not managed by this class.
  bool coefficient_is_variable;
  bool consider_inverse_coefficient;

  VariableCoefficients<dealii::VectorizedArray<Number>> const * variable_coefficients;

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

/*
 * Inverse mass operator for H(div)-conforming space:
 *
 * This class applies the inverse mass operator by solving the mass system as a global linear system
 * of equations for all degrees of freedom. It is used in case the mass operator is not
 * block-diagonal and can not be inverted element-wise (e.g. H(div)-conforming space).
 */
// template<int dim, int n_components, typename Number>
// class InverseMassOperatorHdiv
// {
// private:
//   typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

// public:
//   void
//   initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
//              dealii::AffineConstraints<Number> const & constraints,
//              InverseMassOperatorDataHdiv const         inverse_mass_operator_data)
//   {
//     // mass operator
//     MassOperatorData<dim> mass_operator_data;
//     mass_operator_data.dof_index  = inverse_mass_operator_data.dof_index;
//     mass_operator_data.quad_index = inverse_mass_operator_data.quad_index;
//     mass_operator.initialize(matrix_free, constraints, mass_operator_data);

//     solver_control =
//       dealii::ReductionControl(inverse_mass_operator_data.parameters.solver_data.max_iter,
//                                inverse_mass_operator_data.parameters.solver_data.abs_tol,
//                                inverse_mass_operator_data.parameters.solver_data.rel_tol);
//     preconditioner_type = inverse_mass_operator_data.parameters.preconditioner;

//     if(preconditioner_type == PreconditionerMass::PointJacobi)
//     {
//       preconditioner =
//         std::make_shared<JacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
//           mass_operator, true /* initialize_preconditioner */);
//     }
//     else if(preconditioner_type == PreconditionerMass::LumpedDiagonal)
//     {
//       VectorType tmp;
//       mass_operator.initialize_dof_vector(tmp);
//       mass_operator.initialize_dof_vector(lumped_diagonal.get_vector());
//       tmp = 1.;
//       mass_operator.vmult(lumped_diagonal.get_vector(), tmp);
//       for(Number & entry : lumped_diagonal.get_vector())
//         if(entry > 1e-30)
//           entry = 1.0 / entry;
//         else
//           entry = 1.;
//     }
//   }

//   /**
//    * This function applies the inverse mass operator. Note that this function allows identical dst,
//    * src vector, i.e. the function can be called like apply(dst, dst).
//    */
//   unsigned int
//   apply(VectorType & dst, VectorType const & src) const
//   {
//     VectorType temp;

//     // Note that the inverse mass operator might be called like inverse_mass.apply(dst, dst),
//     // i.e. with identical destination and source vectors. In this case, we need to make sure
//     // that the result is still correct.
//     const VectorType * src_ptr;
//     if(&dst == &src)
//     {
//       temp    = src;
//       src_ptr = &temp;
//     }
//     else
//     {
//       src_ptr = &src;
//     }

//     dealii::SolverCG<VectorType> solver(solver_control);
//     if(preconditioner_type == PreconditionerMass::None)
//     {
//       solver.solve(mass_operator, dst, *src_ptr, dealii::PreconditionIdentity());
//     }
//     else if(preconditioner_type == PreconditionerMass::PointJacobi)
//     {
//       AssertThrow(preconditioner.get() != nullptr,
//                   dealii::ExcMessage("Preconditioner not initialized!"));
//       solver.solve(mass_operator, dst, *src_ptr, *preconditioner);
//     }
//     else if(preconditioner_type == PreconditionerMass::LumpedDiagonal)
//     {
//       solver.solve(mass_operator, dst, *src_ptr, lumped_diagonal);
//     }
//     else
//       AssertThrow(false,
//                   dealii::ExcMessage(
//                     "Preconditioner type for Hdiv inverse mass matrix not recognized"));

//     return solver_control.last_step();
//   }

// private:
//   // Solver/preconditioner for mass system solving a global linear system of equations for all
//   // degrees of freedom.
//   std::shared_ptr<PreconditionerBase<Number>> preconditioner;
//   dealii::DiagonalMatrix<VectorType>          lumped_diagonal;
//   dealii::ReductionControl mutable solver_control;

//   // We need a MassOperator as underlying operator.
//   MassOperator<dim, n_components, Number> mass_operator;

//   PreconditionerMass preconditioner_type;
// };

} // namespace ExaDG

#endif /* EXADG_OPERATORS_INVERSE_MASS_OPERATOR_H_ */
