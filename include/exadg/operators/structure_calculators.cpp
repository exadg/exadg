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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/operators/structure_calculators.h>
#include <exadg/structure/material/material_handler.h>

namespace ExaDG
{
template<int dim, typename Number>
DisplacementJacobianCalculator<dim, Number>::DisplacementJacobianCalculator()
  : matrix_free(nullptr), dof_index_vector(0), dof_index_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_vector_in,
  unsigned int const                      dof_index_scalar_in,
  unsigned int const                      quad_index_in)
{
  matrix_free      = &matrix_free_in;
  dof_index_vector = dof_index_vector_in;
  dof_index_scalar = dof_index_scalar_in;
  quad_index       = quad_index_in;
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_scalar_valued,
  VectorType const & src_vector_valued) const
{
  dst_scalar_valued = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst_scalar_valued, src_vector_valued);
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_scalar_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    // Do not enforce constraints on the `src` vector, as constraints are already applied and the
    // `dealii::MatrixFree` object might store different constraints.
    integrator_vector.read_dof_values_plain(src_vector_valued);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      tensor const gradient_displacement = integrator_vector.get_gradient(q);
      tensor const F                     = Structure::get_F(gradient_displacement);
      scalar const Jacobian              = determinant(F);

      integrator_scalar.submit_value(Jacobian, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst_scalar_valued);
  }
}

template<int dim, typename Number>
MaxPrincipalStressCalculator<dim, Number>::MaxPrincipalStressCalculator()
  : matrix_free(nullptr),
    dof_index_vector(0),
    dof_index_scalar(0),
    quad_index(0),
    elasticity_operator_base(nullptr)
{
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  unsigned int const                                     dof_index_vector_in,
  unsigned int const                                     dof_index_scalar_in,
  unsigned int const                                     quad_index_in,
  Structure::ElasticityOperatorBase<dim, Number> const & elasticity_operator_base_in)
{
  matrix_free              = &matrix_free_in;
  dof_index_vector         = dof_index_vector_in;
  dof_index_scalar         = dof_index_scalar_in;
  quad_index               = quad_index_in;
  elasticity_operator_base = &elasticity_operator_base_in;
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_scalar_valued,
  VectorType const & src_vector_valued) const
{
  dst_scalar_valued = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst_scalar_valued, src_vector_valued);
}

template<int dim, typename Number>
void
MaxPrincipalStressCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_scalar_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // material_handler->reinit(matrix_free, cell);
    // std::shared_ptr<Material<dim, Number>> material = material_handler.get_material();

    Structure::Material<dim, Number> const & material =
      elasticity_operator_base->get_material_in_cell(matrix_free, cell);

    integrator_vector.reinit(cell);
    // Do not enforce constraints on the `src` vector, as constraints are already applied and the
    // `dealii::MatrixFree` object might store different constraints.
    integrator_vector.read_dof_values_plain(src_vector_valued);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      tensor const           gradient_displacement = integrator_vector.get_gradient(q);
      tensor const           F                     = Structure::get_F(gradient_displacement);
      scalar const           Jacobian              = determinant(F);
      symmetric_tensor const S =
        material.second_piola_kirchhoff_stress(gradient_displacement, cell, q);
      symmetric_tensor const sigma = Structure::compute_push_forward(Jacobian, S, F);

      // Loop over vectorization length to use `dealii::eigenvalues()`.
      scalar max_eigenvalue;
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); v++)
      {
        dealii::SymmetricTensor<2, dim, Number> sigma_v;
        for(unsigned int i = 0; i < dim; ++i)
        {
          for(unsigned int j = 0; j < dim; ++j)
          {
            sigma_v[i][j] = sigma[i][j][v];
          }
        }
        std::array<Number, dim> eigenvalues_v = dealii::eigenvalues(sigma_v);

        max_eigenvalue[v] = eigenvalues_v[0];
      }

      integrator_scalar.submit_value(max_eigenvalue, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst_scalar_valued);
  }
}

template class DisplacementJacobianCalculator<2, float>;
template class DisplacementJacobianCalculator<2, double>;

template class DisplacementJacobianCalculator<3, float>;
template class DisplacementJacobianCalculator<3, double>;

template class MaxPrincipalStressCalculator<2, float>;
template class MaxPrincipalStressCalculator<2, double>;

template class MaxPrincipalStressCalculator<3, float>;
template class MaxPrincipalStressCalculator<3, double>;

} // namespace ExaDG
