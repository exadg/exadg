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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
StVenantKirchhoff<dim, Number>::StVenantKirchhoff(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  StVenantKirchhoffData<dim> const &      data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    E_is_variable(data.E_function != nullptr)
{
  Number const E = data.E;
  f0             = dealii::make_vectorized_array<Number>(get_f0_factor() * E);
  f1             = dealii::make_vectorized_array<Number>(get_f1_factor() * E);
  f2             = dealii::make_vectorized_array<Number>(get_f2_factor() * E);

  if(E_is_variable)
  {
    // allocate vectors for variable coefficients and initialize with constant values
    f0_coefficients.initialize(matrix_free, quad_index, false, false, false, get_f0_factor() * E);
    f1_coefficients.initialize(matrix_free, quad_index, false, false, false, get_f1_factor() * E);
    f2_coefficients.initialize(matrix_free, quad_index, false, false, false, get_f2_factor() * E);

    VectorType dummy;
    matrix_free.cell_loop(&StVenantKirchhoff<dim, Number>::cell_loop_set_coefficients,
                          this,
                          dummy,
                          dummy);
  }
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f0_factor() const
{
  Number const nu           = data.nu;
  Type2D const type_two_dim = data.type_two_dim;

  return (dim == 3) ?
           (1. - nu) / (1. + nu) / (1. - 2. * nu) :
           (type_two_dim == Type2D::PlaneStress ? (1. / (1. - nu * nu)) :
                                                  ((1. - nu) / (1. + nu) / (1. - 2. * nu)));
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f1_factor() const
{
  Number const nu           = data.nu;
  Type2D const type_two_dim = data.type_two_dim;

  return (dim == 3) ? (nu) / (1. + nu) / (1. - 2. * nu) :
                      (type_two_dim == Type2D::PlaneStress ? (nu / (1. - nu * nu)) :
                                                             (nu / (1. + nu) / (1. - 2. * nu)));
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f2_factor() const
{
  Number const nu           = data.nu;
  Type2D const type_two_dim = data.type_two_dim;

  return (dim == 3) ? (1. - 2. * nu) / 2. / (1. + nu) / (1. - 2. * nu) :
                      (type_two_dim == Type2D::PlaneStress ?
                         ((1. - nu) / 2. / (1. - nu * nu)) :
                         ((1. - 2. * nu) / 2. / (1. + nu) / (1. - 2. * nu)));
}

template<int dim, typename Number>
void
StVenantKirchhoff<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const &,
  Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::VectorizedArray<Number> E_vec =
        FunctionEvaluator<0, dim, Number>::value(data.E_function,
                                                 integrator.quadrature_point(q),
                                                 0.0 /*time*/);

      // set the coefficients
      f0_coefficients.set_coefficient_cell(cell, q, get_f0_factor() * E_vec);
      f1_coefficients.set_coefficient_cell(cell, q, get_f1_factor() * E_vec);
      f2_coefficients.set_coefficient_cell(cell, q, get_f2_factor() * E_vec);
    }
  }
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
StVenantKirchhoff<dim, Number>::evaluate_stress(
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & E,
  unsigned int const                                              cell,
  unsigned int const                                              q) const
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> S;

  if(E_is_variable)
  {
    f0 = f0_coefficients.get_coefficient_cell(cell, q);
    f1 = f1_coefficients.get_coefficient_cell(cell, q);
    f2 = f2_coefficients.get_coefficient_cell(cell, q);
  }

  if(dim == 3)
  {
    S[0][0] = f0 * E[0][0] + f1 * E[1][1] + f1 * E[2][2];
    S[1][1] = f1 * E[0][0] + f0 * E[1][1] + f1 * E[2][2];
    S[2][2] = f1 * E[0][0] + f1 * E[1][1] + f0 * E[2][2];
    S[0][1] = f2 * (E[0][1] + E[1][0]);
    S[1][2] = f2 * (E[1][2] + E[2][1]);
    S[0][2] = f2 * (E[0][2] + E[2][0]);
    S[1][0] = f2 * (E[0][1] + E[1][0]);
    S[2][1] = f2 * (E[1][2] + E[2][1]);
    S[2][0] = f2 * (E[0][2] + E[2][0]);
  }
  else
  {
    S[0][0] = f0 * E[0][0] + f1 * E[1][1];
    S[1][1] = f1 * E[0][0] + f0 * E[1][1];
    S[0][1] = f2 * (E[0][1] + E[1][0]);
    S[1][0] = f2 * (E[0][1] + E[1][0]);
  }

  return S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
StVenantKirchhoff<dim, Number>::apply_C(
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & E,
  unsigned int const                                              cell,
  unsigned int const                                              q) const
{
  return evaluate_stress(E, cell, q);
}

template class StVenantKirchhoff<2, float>;
template class StVenantKirchhoff<2, double>;

template class StVenantKirchhoff<3, float>;
template class StVenantKirchhoff<3, double>;

} // namespace Structure
} // namespace ExaDG
