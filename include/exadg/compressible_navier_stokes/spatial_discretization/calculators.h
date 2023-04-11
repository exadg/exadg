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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace CompNS
{
/*
 *  The DoF vector contains the vector of conserved quantities (rho, rho u, rho E).
 *  This class allows to transfer these quantities into the derived variables (p, u, T)
 *  by using L2-projections.
 */
template<int dim, typename Number>
class p_u_T_Calculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef p_u_T_Calculator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  p_u_T_Calculator()
    : matrix_free(nullptr),
      dof_index_all(0),
      dof_index_vector(1),
      dof_index_scalar(2),
      quad_index(0),
      heat_capacity_ratio(-1.0),
      specific_gas_constant(-1.0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_all_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in,
             double const                            heat_capacity_ratio_in,
             double const                            specific_gas_constant_in)
  {
    matrix_free           = &matrix_free_in;
    dof_index_all         = dof_index_all_in;
    dof_index_vector      = dof_index_vector_in;
    dof_index_scalar      = dof_index_scalar_in;
    quad_index            = quad_index_in;
    heat_capacity_ratio   = heat_capacity_ratio_in;
    specific_gas_constant = specific_gas_constant_in;
  }

  void
  compute_pressure(VectorType & pressure, VectorType const & solution_conserved) const
  {
    AssertThrow(heat_capacity_ratio > 0.0,
                dealii::ExcMessage("heat capacity ratio has not been set!"));
    AssertThrow(specific_gas_constant > 0.0,
                dealii::ExcMessage("specific gas constant has not been set!"));

    matrix_free->cell_loop(&This::local_apply_pressure, this, pressure, solution_conserved);
  }

  void
  compute_velocity(VectorType & velocity, VectorType const & solution_conserved) const
  {
    matrix_free->cell_loop(&This::local_apply_velocity, this, velocity, solution_conserved);
  }

  void
  compute_temperature(VectorType & temperature, VectorType const & solution_conserved) const
  {
    AssertThrow(heat_capacity_ratio > 0.0,
                dealii::ExcMessage("heat capacity ratio has not been set!"));
    AssertThrow(specific_gas_constant > 0.0,
                dealii::ExcMessage("specific gas constant has not been set!"));

    matrix_free->cell_loop(&This::local_apply_temperature, this, temperature, solution_conserved);
  }

private:
  void
  local_apply_pressure(dealii::MatrixFree<dim, Number> const &       matrix_free,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);
    CellIntegratorScalar energy(matrix_free, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    CellIntegratorScalar pressure(matrix_free, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(dealii::EvaluationFlags::values);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(dealii::EvaluationFlags::values);
      energy.reinit(cell);
      energy.read_dof_values(src);
      energy.evaluate(dealii::EvaluationFlags::values);

      // dst-vector
      pressure.reinit(cell);

      for(unsigned int q = 0; q < energy.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);
        scalar rho_E = energy.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;

        scalar p = dealii::make_vectorized_array<Number>(heat_capacity_ratio - 1.0) *
                   (rho_E - 0.5 * rho * scalar_product(u, u));

        pressure.submit_value(p, q);
      }

      pressure.integrate(dealii::EvaluationFlags::values);
      pressure.set_dof_values(dst);
    }
  }

  void
  local_apply_velocity(dealii::MatrixFree<dim, Number> const &       matrix_free,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);

    // dst-vector
    CellIntegratorVector velocity(matrix_free, dof_index_vector, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(dealii::EvaluationFlags::values);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(dealii::EvaluationFlags::values);

      // dst-vector
      velocity.reinit(cell);

      for(unsigned int q = 0; q < momentum.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;

        velocity.submit_value(u, q);
      }

      velocity.integrate(dealii::EvaluationFlags::values);
      velocity.set_dof_values(dst);
    }
  }

  void
  local_apply_temperature(dealii::MatrixFree<dim, Number> const &       matrix_free,
                          VectorType &                                  dst,
                          VectorType const &                            src,
                          std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);
    CellIntegratorScalar energy(matrix_free, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    CellIntegratorScalar temperature(matrix_free, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(dealii::EvaluationFlags::values);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(dealii::EvaluationFlags::values);
      energy.reinit(cell);
      energy.read_dof_values(src);
      energy.evaluate(dealii::EvaluationFlags::values);

      // dst-vector
      temperature.reinit(cell);

      for(unsigned int q = 0; q < energy.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);
        scalar rho_E = energy.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;
        scalar E = rho_E / rho;

        scalar T = dealii::make_vectorized_array<Number>((heat_capacity_ratio - 1.0) /
                                                         specific_gas_constant) *
                   (E - 0.5 * scalar_product(u, u));

        temperature.submit_value(T, q);
      }

      temperature.integrate(dealii::EvaluationFlags::values);
      temperature.set_dof_values(dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_all;
  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;

  double heat_capacity_ratio;
  double specific_gas_constant;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_H_ \
        */
