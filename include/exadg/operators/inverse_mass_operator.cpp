/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

#include <exadg/operators/inverse_mass_operator.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
InverseMassOperator<dim, n_components, Number>::InverseMassOperator()
  : matrix_free(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, int n_components, typename Number>
unsigned int
InverseMassOperator<dim, n_components, Number>::get_n_iter_global_last() const
{
  return this->n_iter_global_last;
}

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free_in,
  InverseMassOperatorData<Number> const     inverse_mass_operator_data,
  dealii::AffineConstraints<Number> const * constraints)
{
  matrix_free = &matrix_free_in;
  dof_index   = inverse_mass_operator_data.dof_index;
  quad_index  = inverse_mass_operator_data.quad_index;

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
      fe.conforms(dealii::FiniteElementData<dim>::L2),
      dealii::ExcMessage(
        "The matrix-free cell-wise inverse mass operator is only available for L2-conforming elements."));

    if(fe.reference_cell().is_hyper_cube())
    {
      AssertThrow(
        fe.base_element(0).dofs_per_cell == dealii::Utilities::pow(fe.degree + 1, dim),
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator is currently only available for isotropic tensor-product elements."));

      AssertThrow(
        this->matrix_free->get_shape_info(0, quad_index).data[0].n_q_points_1d == fe.degree + 1,
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator is currently only available if n_q_points_1d = n_nodes_1d."));

      is_hypercube_element = true;
    }
    else if(fe.reference_cell().is_simplex())
    {
      const bool cartesian_or_affine_mapping =
        std::all_of(matrix_free->get_mapping_info().cell_type.begin(),
                    matrix_free->get_mapping_info().cell_type.end(),
                    [](auto g) {
                      return g <= dealii::internal::MatrixFreeFunctions::GeometryType::affine;
                    });
      AssertThrow(cartesian_or_affine_mapping,
                  dealii::ExcMessage(
                    "The matrix-free cell-wise inverse mass operator can only be applied "
                    "with affine mapping on non-hypercube elements."));

      AssertThrow(
        !coefficient_is_variable,
        dealii::ExcMessage(
          "The matrix-free cell-wise inverse mass operator can only be applied with constant "
          "coefficients over a cell on non-hypercube elements, use apply_scale() in this case."));

      is_hypercube_element = false;
      // setup mass matrix on reference element
      {
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::reference_cell(triangulation, fe.reference_cell());
        const dealii::FE_SimplexDGP<dim> scalar_fe(fe.degree);

        const unsigned int         dofs_per_cell = scalar_fe.n_dofs_per_cell();
        dealii::FullMatrix<Number> mass_matrix(dofs_per_cell, dofs_per_cell);

        const dealii::MappingFE<dim, dim> mapping_mass(scalar_fe);
        const auto                        quadrature_mass =
          scalar_fe.reference_cell().template get_gauss_type_quadrature<dim>(scalar_fe.degree + 1);

        dealii::FEValues<dim> fe_values(mapping_mass,
                                        scalar_fe,
                                        quadrature_mass,
                                        dealii::update_values | dealii::update_quadrature_points |
                                          dealii::update_JxW_values);
        fe_values.reinit(triangulation.begin_active());

        mass_matrix = 0.;
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            for(unsigned int q = 0; q < quadrature_mass.size(); ++q)
              mass_matrix(i, j) +=
                fe_values.JxW(q) * fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
        mass_matrix.gauss_jordan();

        this->inverse_mass_matrix.reserve(dofs_per_cell * dofs_per_cell);
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            this->inverse_mass_matrix.emplace_back(mass_matrix(i, j));
      }
    }
    else
      DEAL_II_NOT_IMPLEMENTED();
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
      if(data.preconditioner == PreconditionerMass::None)
      {
        mass_operator_data.preconditioner_block_diagonal = Elementwise::Preconditioner::None;
      }
      else if(data.preconditioner == PreconditionerMass::PointJacobi)
      {
        mass_operator_data.preconditioner_block_diagonal = Elementwise::Preconditioner::PointJacobi;
      }
      mass_operator_data.solver_data_block_diagonal = data.solver_data;
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

      // Store `0` to signal that no global iterations are done.
      this->n_iter_global_last = 0;
    }
    else if(data.implementation_type == InverseMassType::GlobalKrylovSolver)
    {
      if(data.preconditioner == PreconditionerMass::None)
      {
        // no setup required.
      }
      else if(data.preconditioner == PreconditionerMass::PointJacobi)
      {
        global_preconditioner =
          std::make_shared<JacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
            mass_operator, true /* initialize_preconditioner */);
      }
      else if(data.preconditioner == PreconditionerMass::BlockJacobi)
      {
        global_preconditioner =
          std::make_shared<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
            mass_operator, true /* initialize_preconditioner */);
      }
      else if(data.preconditioner == PreconditionerMass::AMG)
      {
        global_preconditioner =
          std::make_shared<PreconditionerAMG<MassOperator<dim, n_components, Number>, Number>>(
            mass_operator, true /* initialize_preconditioner */, data.amg_data);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("This `PreconditionerMass` is not implemented."));
      }

      bool constexpr compute_performance_metrics = false;
      bool constexpr compute_eigenvalues         = false;
      bool const use_preconditioner              = data.preconditioner != PreconditionerMass::None;

      typedef Krylov::KrylovSolver<MassOperator<dim, n_components, Number>,
                                   PreconditionerBase<Number>,
                                   VectorType>
        SolverType;

      global_solver = std::make_shared<SolverType>(mass_operator,
                                                   *global_preconditioner,
                                                   data.solver_data,
                                                   use_preconditioner,
                                                   compute_performance_metrics,
                                                   compute_eigenvalues);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified `InverseMassType` is not implemented."));
    }
  }
}

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::update()
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

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::apply(VectorType &       dst,
                                                      VectorType const & src) const
{
  if(data.implementation_type == InverseMassType::GlobalKrylovSolver)
  {
    AssertThrow(global_solver.get() != 0,
                dealii::ExcMessage("Global mass solver has not been initialized."));
    this->n_iter_global_last = global_solver->solve(dst, src);
  }
  else
  {
    dst.zero_out_ghost_values();

    if(data.implementation_type == InverseMassType::MatrixfreeOperator && is_hypercube_element)
    {
      matrix_free->cell_loop(&This::cell_loop_matrix_free_operator, this, dst, src);
    }
    else if(data.implementation_type == InverseMassType::MatrixfreeOperator &&
            !is_hypercube_element)
    {
      matrix_free->cell_loop(&This::cell_loop_matrix_free_operator_simplex, this, dst, src);
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

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::apply_scale(VectorType &       dst,
                                                            double const       scaling_factor,
                                                            VectorType const & src) const
{
  if(data.implementation_type == InverseMassType::MatrixfreeOperator && is_hypercube_element)
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
  else if(data.implementation_type == InverseMassType::MatrixfreeOperator && !is_hypercube_element)
  {
    // In the InverseMassType::MatrixfreeOperator case we can avoid
    // streaming the vector from memory twice.

    // ghost entries have to be zeroed out before `MatrixFree::cell_loop()`.
    dst.zero_out_ghost_values();

    matrix_free->cell_loop(
      &This::cell_loop_matrix_free_operator_simplex,
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

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::cell_loop_matrix_free_operator(
  dealii::MatrixFree<dim, Number> const &,
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

template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::cell_loop_matrix_free_operator_simplex(
  dealii::MatrixFree<dim, Number> const &,
  VectorType &       dst,
  VectorType const & src,
  Range const &      cell_range) const
{
  Integrator integrator(*matrix_free, dof_index, quad_index);

  const unsigned int dofs_per_cell = integrator.dofs_per_component;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> values_dofs_inverse(dofs_per_cell *
                                                                             n_components);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    // Apply inverse mass on components
    for(unsigned int c = 0; c < n_components; ++c)
    {
      dealii::internal::apply_matrix_vector_product<
        dealii::internal::EvaluatorVariant::evaluate_general,
        dealii::internal::EvaluatorQuantity::value,
        /*transpose_matrix*/ false,
        /*add*/ false,
        /*consider_strides*/ false>(inverse_mass_matrix.data(),
                                    integrator.begin_dof_values() + c * dofs_per_cell,
                                    &values_dofs_inverse[0] + c * dofs_per_cell,
                                    dofs_per_cell,
                                    dofs_per_cell,
                                    1,
                                    1);
    }

    // apply inverse jacobi matrix
    const auto &       mapping_data = matrix_free->get_mapping_info().cell_data[quad_index];
    const unsigned int offsets      = mapping_data.data_index_offsets[cell];
    const dealii::VectorizedArray<Number> * j_value = &mapping_data.JxW_values[offsets];

    const dealii::VectorizedArray<Number> j_inverse = 1. / j_value[0];
    for(unsigned int i = 0; i < n_components * dofs_per_cell; ++i)
      integrator.begin_dof_values()[i] = values_dofs_inverse[i] * j_inverse;

    integrator.set_dof_values(dst);
  }
}

template class InverseMassOperator<2, 1, float>;
template class InverseMassOperator<2, 1, double>;

template class InverseMassOperator<3, 1, float>;
template class InverseMassOperator<3, 1, double>;

template class InverseMassOperator<2, 2, float>;
template class InverseMassOperator<2, 2, double>;

template class InverseMassOperator<3, 3, float>;
template class InverseMassOperator<3, 3, double>;

template class InverseMassOperator<2, 4, float>;
template class InverseMassOperator<2, 4, double>;

template class InverseMassOperator<3, 5, float>;
template class InverseMassOperator<3, 5, double>;

} // namespace ExaDG
