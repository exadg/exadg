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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_KERNELS_AND_OPERATORS_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_KERNELS_AND_OPERATORS_H_

// C/C++
#include <iostream>

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/compressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/interior_penalty_parameter.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<Number>
    calculate_pressure(dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & rho_u,
                       dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u,
                       dealii::VectorizedArray<Number> const &                         rho_E,
                       Number const &                                                  gamma)
{
  return (gamma - 1.0) * (rho_E - 0.5 * scalar_product(rho_u, u));
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  calculate_pressure(dealii::VectorizedArray<Number> const &                         rho,
                     dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u,
                     dealii::VectorizedArray<Number> const &                         E,
                     Number const &                                                  gamma)
{
  return (gamma - 1.0) * rho * (E - 0.5 * scalar_product(u, u));
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  calculate_temperature(dealii::VectorizedArray<Number> const & p,
                        dealii::VectorizedArray<Number> const & rho,
                        Number const &                          R)
{
  return p / (rho * R);
}

template<int dim, typename Number>
inline dealii::VectorizedArray<Number>
calculate_energy(dealii::VectorizedArray<Number> const &                         T,
                 dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u,
                 Number const &                                                  c_v)
{
  return c_v * T + 0.5 * scalar_product(u, u);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
  calculate_grad_E(dealii::VectorizedArray<Number> const &                         rho_inverse,
                   dealii::VectorizedArray<Number> const &                         rho_E,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & grad_rho,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & grad_rho_E)
{
  dealii::VectorizedArray<Number> E = rho_inverse * rho_E;

  return rho_inverse * (grad_rho_E - E * grad_rho);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  calculate_grad_u(dealii::VectorizedArray<Number> const &                         rho_inverse,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & rho_u,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & grad_rho,
                   dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & grad_rho_u)
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> out;
  for(unsigned int d = 0; d < dim; ++d)
  {
    dealii::VectorizedArray<Number> ud = rho_inverse * rho_u[d];
    for(unsigned int e = 0; e < dim; ++e)
      out[d][e] = rho_inverse * (grad_rho_u[d][e] - ud * grad_rho[e]);
  }

  return out;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    calculate_grad_T(dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & grad_E,
                     dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u,
                     dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & grad_u,
                     Number const &                                                  gamma,
                     Number const &                                                  R)
{
  return (gamma - 1.0) / R * (grad_E - u * grad_u);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    calculate_stress_tensor(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & grad_u,
                            Number const & viscosity)
{
  dealii::VectorizedArray<Number> const divu = (2. / 3.) * trace(grad_u);

  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> out;
  for(unsigned int d = 0; d < dim; ++d)
  {
    for(unsigned int e = 0; e < dim; ++e)
      out[d][e] = viscosity * (grad_u[d][e] + grad_u[e][d]);
    out[d][d] -= viscosity * divu;
  }

  return out;
}

/*
 * Calculates exterior state "+" for a scalar/vectorial quantity depending on interior state "-" and
 * boundary conditions.
 */
template<int dim, typename Number, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_exterior_value(
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> const & value_m,
    BoundaryType const                                                 boundary_type,
    BoundaryDescriptorStd<dim> const &                                 boundary_descriptor,
    dealii::types::boundary_id const &                                 boundary_id,
    dealii::Point<dim, dealii::VectorizedArray<Number>> const &        q_point,
    Number const &                                                     time)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> value_p;

  if(boundary_type == BoundaryType::Dirichlet)
  {
    auto bc = boundary_descriptor.dirichlet_bc.find(boundary_id)->second;
    auto g  = FunctionEvaluator<rank, dim, Number>::value(bc, q_point, time);

    value_p = -value_m + dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>(2.0 * g);
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return value_p;
}

/*
 * Calculates exterior state of normal gradient (Neumann type boundary conditions)
 * depending on interior data and boundary conditions.
 */
template<int dim, typename Number, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_exterior_normal_grad(
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> const & grad_M_normal,
    BoundaryType const &                                               boundary_type,
    BoundaryDescriptorStd<dim> const &                                 boundary_descriptor,
    dealii::types::boundary_id const &                                 boundary_id,
    dealii::Point<dim, dealii::VectorizedArray<Number>> const &        q_point,
    Number const &                                                     time)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> grad_P_normal;

  if(boundary_type == BoundaryType::Dirichlet)
  {
    // do nothing
    grad_P_normal = grad_M_normal;
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    auto bc = boundary_descriptor.neumann_bc.find(boundary_id)->second;
    auto h  = FunctionEvaluator<rank, dim, Number>::value(bc, q_point, time);

    grad_P_normal =
      -grad_M_normal + dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>(2.0 * h);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return grad_P_normal;
}

/*
 * This function calculates the Lax-Friedrichs flux for the momentum equation
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    calculate_flux(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & momentum_flux_M,
                   dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & momentum_flux_P,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & rho_u_M,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & rho_u_P,
                   dealii::VectorizedArray<Number> const &                         lambda,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normal)
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> out;
  for(unsigned int d = 0; d < dim; ++d)
  {
    dealii::VectorizedArray<Number> sum = dealii::VectorizedArray<Number>();
    for(unsigned int e = 0; e < dim; ++e)
      sum += (momentum_flux_M[d][e] + momentum_flux_P[d][e]) * normal[e];
    out[d] = 0.5 * (sum + lambda * (rho_u_M[d] - rho_u_P[d]));
  }

  return out;
}

/*
 * This function calculates the Lax-Friedrichs flux for scalar quantities (density/energy)
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    dealii::VectorizedArray<Number>
    calculate_flux(dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & flux_M,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & flux_P,
                   dealii::VectorizedArray<Number> const &                         value_M,
                   dealii::VectorizedArray<Number> const &                         value_P,
                   dealii::VectorizedArray<Number> const &                         lambda,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normal)
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> average_flux = 0.5 * (flux_M + flux_P);

  return average_flux * normal + 0.5 * lambda * (value_M - value_P);
}

/*
 * Calculation of lambda for Lax-Friedrichs flux according to Hesthaven:
 *   lambda = max( |u_M| + sqrt(|gamma * p_M / rho_M|) , |u_P| + sqrt(|gamma * p_P / rho_P|) )
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  calculate_lambda(dealii::VectorizedArray<Number> const &                         rho_m,
                   dealii::VectorizedArray<Number> const &                         rho_p,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u_m,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & u_p,
                   dealii::VectorizedArray<Number> const &                         p_m,
                   dealii::VectorizedArray<Number> const &                         p_p,
                   Number const &                                                  gamma)
{
  dealii::VectorizedArray<Number> lambda_m = u_m.norm() + std::sqrt(std::abs(gamma * p_m / rho_m));
  dealii::VectorizedArray<Number> lambda_p = u_p.norm() + std::sqrt(std::abs(gamma * p_p / rho_p));

  return std::max(lambda_m, lambda_p);
}

template<int dim>
struct BodyForceOperatorData
{
  BodyForceOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<dealii::Function<dim>> rhs_rho;
  std::shared_ptr<dealii::Function<dim>> rhs_u;
  std::shared_ptr<dealii::Function<dim>> rhs_E;
};

template<int dim, typename Number>
class BodyForceOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef BodyForceOperator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;
  typedef dealii::Point<dim, dealii::VectorizedArray<Number>>     point;

  BodyForceOperator() : matrix_free(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             BodyForceOperatorData<dim> const &      data_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;
  }

  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, vector, scalar>
    get_volume_flux(CellIntegratorScalar & density,
                    CellIntegratorVector & momentum,
                    unsigned int const     q) const
  {
    point  q_points = density.quadrature_point(q);
    scalar rho      = density.get_value(q);
    vector u        = momentum.get_value(q) / rho;

    scalar rhs_density =
      FunctionEvaluator<0, dim, Number>::value(data.rhs_rho, q_points, eval_time);
    vector rhs_momentum = FunctionEvaluator<1, dim, Number>::value(data.rhs_u, q_points, eval_time);
    scalar rhs_energy   = FunctionEvaluator<0, dim, Number>::value(data.rhs_E, q_points, eval_time);

    return std::make_tuple(rhs_density, rhs_momentum, rhs_momentum * u + rhs_energy);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar density(matrix_free, data.dof_index, data.quad_index, 0);
    CellIntegratorVector momentum(matrix_free, data.dof_index, data.quad_index, 1);
    CellIntegratorScalar energy(matrix_free, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      density.reinit(cell);
      density.gather_evaluate(src, dealii::EvaluationFlags::values);

      momentum.reinit(cell);
      momentum.gather_evaluate(src, dealii::EvaluationFlags::values);

      energy.reinit(cell);

      for(unsigned int q = 0; q < density.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> flux = get_volume_flux(density, momentum, q);

        density.submit_value(std::get<0>(flux), q);
        momentum.submit_value(std::get<1>(flux), q);
        energy.submit_value(std::get<2>(flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::values, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  BodyForceOperatorData<dim> data;

  double mutable eval_time;
};

struct MassOperatorData
{
  MassOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, typename Number>
class MassOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef MassOperator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  MassOperator() : matrix_free(nullptr)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             MassOperatorData const &                data_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst = 0;
    apply_add(dst, src);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar density(matrix_free, data.dof_index, data.quad_index, 0);
    CellIntegratorVector momentum(matrix_free, data.dof_index, data.quad_index, 1);
    CellIntegratorScalar energy(matrix_free, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      density.reinit(cell);
      density.gather_evaluate(src, dealii::EvaluationFlags::values);

      momentum.reinit(cell);
      momentum.gather_evaluate(src, dealii::EvaluationFlags::values);

      energy.reinit(cell);
      energy.gather_evaluate(src, dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < density.n_q_points; ++q)
      {
        density.submit_value(density.get_value(q), q);
        momentum.submit_value(momentum.get_value(q), q);
        energy.submit_value(energy.get_value(q), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::values, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;
  MassOperatorData                        data;
};

template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData()
    : dof_index(0), quad_index(0), heat_capacity_ratio(1.4), specific_gas_constant(287.0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;

  double heat_capacity_ratio;
  double specific_gas_constant;
};

template<int dim, typename Number>
class ConvectiveOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ConvectiveOperator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorVector;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;
  typedef dealii::Point<dim, dealii::VectorizedArray<Number>>     point;

  ConvectiveOperator() : matrix_free(nullptr)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             ConvectiveOperatorData<dim> const &     data_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;

    gamma = data.heat_capacity_ratio;
    R     = data.specific_gas_constant;
    c_v   = R / (gamma - 1.0);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

  void
  set_evaluation_time(double const & evaluation_time) const
  {
    eval_time = evaluation_time;
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, tensor, vector>
    get_volume_flux(CellIntegratorScalar & density,
                    CellIntegratorVector & momentum,
                    CellIntegratorScalar & energy,
                    unsigned int const     q) const
  {
    scalar rho_inv = 1.0 / density.get_value(q);
    vector rho_u   = momentum.get_value(q);
    scalar rho_E   = energy.get_value(q);
    vector u       = rho_inv * rho_u;
    scalar p       = calculate_pressure(rho_u, u, rho_E, gamma);

    tensor momentum_flux = outer_product(rho_u, u);
    for(unsigned int d = 0; d < dim; ++d)
      momentum_flux[d][d] += p;

    vector energy_flux = (rho_E + p) * u;

    return std::make_tuple(rho_u, momentum_flux, energy_flux);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, vector, scalar>
    get_flux(FaceIntegratorScalar & density_m,
             FaceIntegratorScalar & density_p,
             FaceIntegratorVector & momentum_m,
             FaceIntegratorVector & momentum_p,
             FaceIntegratorScalar & energy_m,
             FaceIntegratorScalar & energy_p,
             unsigned int const     q) const
  {
    vector normal = momentum_m.get_normal_vector(q);

    // get values
    scalar rho_M   = density_m.get_value(q);
    scalar rho_P   = density_p.get_value(q);
    vector rho_u_M = momentum_m.get_value(q);
    vector rho_u_P = momentum_p.get_value(q);
    vector u_M     = rho_u_M / rho_M;
    vector u_P     = rho_u_P / rho_P;
    scalar rho_E_M = energy_m.get_value(q);
    scalar rho_E_P = energy_p.get_value(q);

    // calculate pressure
    scalar p_M = calculate_pressure(rho_u_M, u_M, rho_E_M, gamma);
    scalar p_P = calculate_pressure(rho_u_P, u_P, rho_E_P, gamma);

    // calculate lambda
    scalar lambda = calculate_lambda(rho_M, rho_P, u_M, u_P, p_M, p_P, gamma);

    // flux density
    scalar flux_density = calculate_flux(rho_u_M, rho_u_P, rho_M, rho_P, lambda, normal);

    // flux momentum
    tensor momentum_flux_M = outer_product(rho_u_M, u_M);
    for(unsigned int d = 0; d < dim; ++d)
      momentum_flux_M[d][d] += p_M;

    tensor momentum_flux_P = outer_product(rho_u_P, u_P);
    for(unsigned int d = 0; d < dim; ++d)
      momentum_flux_P[d][d] += p_P;

    vector flux_momentum =
      calculate_flux(momentum_flux_M, momentum_flux_P, rho_u_M, rho_u_P, lambda, normal);

    // flux energy
    vector energy_flux_M = (rho_E_M + p_M) * u_M;
    vector energy_flux_P = (rho_E_P + p_P) * u_P;

    scalar flux_energy =
      calculate_flux(energy_flux_M, energy_flux_P, rho_E_M, rho_E_P, lambda, normal);

    return std::make_tuple(flux_density, flux_momentum, flux_energy);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, vector, scalar>
    get_flux_boundary(FaceIntegratorScalar &             density,
                      FaceIntegratorVector &             momentum,
                      FaceIntegratorScalar &             energy,
                      BoundaryType const &               boundary_type_density,
                      BoundaryType const &               boundary_type_velocity,
                      BoundaryType const &               boundary_type_pressure,
                      BoundaryType const &               boundary_type_energy,
                      EnergyBoundaryVariable const &     boundary_variable,
                      dealii::types::boundary_id const & boundary_id,
                      unsigned int const                 q) const
  {
    vector normal = momentum.get_normal_vector(q);

    // element e⁻
    scalar rho_M   = density.get_value(q);
    vector rho_u_M = momentum.get_value(q);
    vector u_M     = rho_u_M / rho_M;
    scalar rho_E_M = energy.get_value(q);
    scalar E_M     = rho_E_M / rho_M;
    scalar p_M     = calculate_pressure(rho_M, u_M, E_M, gamma);

    // element e⁺

    // calculate rho_P
    scalar rho_P = calculate_exterior_value<dim, Number, 0>(rho_M,
                                                            boundary_type_density,
                                                            data.bc->density,
                                                            boundary_id,
                                                            density.quadrature_point(q),
                                                            this->eval_time);

    // calculate u_P
    vector u_P = calculate_exterior_value<dim, Number, 1>(u_M,
                                                          boundary_type_velocity,
                                                          data.bc->velocity,
                                                          boundary_id,
                                                          momentum.quadrature_point(q),
                                                          this->eval_time);

    vector rho_u_P = rho_P * u_P;

    // calculate p_P
    scalar p_P = calculate_exterior_value<dim, Number, 0>(p_M,
                                                          boundary_type_pressure,
                                                          data.bc->pressure,
                                                          boundary_id,
                                                          density.quadrature_point(q),
                                                          this->eval_time);

    // calculate E_P
    scalar E_P = dealii::make_vectorized_array<Number>(0.0);
    if(boundary_variable == EnergyBoundaryVariable::Energy)
    {
      E_P = calculate_exterior_value<dim, Number, 0>(E_M,
                                                     boundary_type_energy,
                                                     data.bc->energy,
                                                     boundary_id,
                                                     energy.quadrature_point(q),
                                                     this->eval_time);
    }
    else if(boundary_variable == EnergyBoundaryVariable::Temperature)
    {
      scalar T_M = calculate_temperature(p_M, rho_M, R);
      scalar T_P = calculate_exterior_value<dim, Number, 0>(T_M,
                                                            boundary_type_energy,
                                                            data.bc->energy,
                                                            boundary_id,
                                                            energy.quadrature_point(q),
                                                            this->eval_time);

      E_P = calculate_energy(T_P, u_P, c_v);
    }
    scalar rho_E_P = rho_P * E_P;

    // calculate lambda
    scalar lambda = calculate_lambda(rho_M, rho_P, u_M, u_P, p_M, p_P, gamma);

    // flux density
    scalar flux_density = calculate_flux(rho_u_M, rho_u_P, rho_M, rho_P, lambda, normal);

    // flux momentum
    tensor momentum_flux_M = outer_product(rho_u_M, u_M);
    for(unsigned int d = 0; d < dim; ++d)
      momentum_flux_M[d][d] += p_M;

    tensor momentum_flux_P = outer_product(rho_u_P, u_P);
    for(unsigned int d = 0; d < dim; ++d)
      momentum_flux_P[d][d] += p_P;

    vector flux_momentum =
      calculate_flux(momentum_flux_M, momentum_flux_P, rho_u_M, rho_u_P, lambda, normal);

    // flux energy
    vector energy_flux_M = (rho_E_M + p_M) * u_M;
    vector energy_flux_P = (rho_E_P + p_P) * u_P;
    scalar flux_energy =
      calculate_flux(energy_flux_M, energy_flux_P, rho_E_M, rho_E_P, lambda, normal);

    return std::make_tuple(flux_density, flux_momentum, flux_energy);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar density(matrix_free, data.dof_index, data.quad_index, 0);
    CellIntegratorVector momentum(matrix_free, data.dof_index, data.quad_index, 1);
    CellIntegratorScalar energy(matrix_free, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      density.reinit(cell);
      density.gather_evaluate(src, dealii::EvaluationFlags::values);

      momentum.reinit(cell);
      momentum.gather_evaluate(src, dealii::EvaluationFlags::values);

      energy.reinit(cell);
      energy.gather_evaluate(src, dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < momentum.n_q_points; ++q)
      {
        std::tuple<vector, tensor, vector> flux = get_volume_flux(density, momentum, energy, q);

        density.submit_gradient(-std::get<0>(flux), q);
        momentum.submit_gradient(-std::get<1>(flux), q);
        energy.submit_gradient(-std::get<2>(flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }
  }

  void
  face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density_m(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorScalar density_p(matrix_free, false, data.dof_index, data.quad_index, 0);

    FaceIntegratorVector momentum_m(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorVector momentum_p(matrix_free, false, data.dof_index, data.quad_index, 1);

    FaceIntegratorScalar energy_m(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);
    FaceIntegratorScalar energy_p(matrix_free, false, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      density_m.reinit(face);
      density_m.gather_evaluate(src, dealii::EvaluationFlags::values);

      density_p.reinit(face);
      density_p.gather_evaluate(src, dealii::EvaluationFlags::values);

      // velocity
      momentum_m.reinit(face);
      momentum_m.gather_evaluate(src, dealii::EvaluationFlags::values);

      momentum_p.reinit(face);
      momentum_p.gather_evaluate(src, dealii::EvaluationFlags::values);

      // energy
      energy_m.reinit(face);
      energy_m.gather_evaluate(src, dealii::EvaluationFlags::values);

      energy_p.reinit(face);
      energy_p.gather_evaluate(src, dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < momentum_m.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> flux =
          get_flux(density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, q);

        density_m.submit_value(std::get<0>(flux), q);
        // - sign since n⁺ = -n⁻
        density_p.submit_value(-std::get<0>(flux), q);

        momentum_m.submit_value(std::get<1>(flux), q);
        // - sign since n⁺ = -n⁻
        momentum_p.submit_value(-std::get<1>(flux), q);

        energy_m.submit_value(std::get<2>(flux), q);
        // - sign since n⁺ = -n⁻
        energy_p.submit_value(-std::get<2>(flux), q);
      }

      density_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      density_p.integrate_scatter(dealii::EvaluationFlags::values, dst);

      momentum_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum_p.integrate_scatter(dealii::EvaluationFlags::values, dst);

      energy_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      energy_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }

  void
  boundary_face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
                     VectorType &                                  dst,
                     VectorType const &                            src,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorVector momentum(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorScalar energy(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      density.reinit(face);
      density.gather_evaluate(src, dealii::EvaluationFlags::values);

      momentum.reinit(face);
      momentum.gather_evaluate(src, dealii::EvaluationFlags::values);

      energy.reinit(face);
      energy.gather_evaluate(src, dealii::EvaluationFlags::values);

      dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      BoundaryType boundary_type_density  = data.bc->density.get_boundary_type(boundary_id);
      BoundaryType boundary_type_velocity = data.bc->velocity.get_boundary_type(boundary_id);
      BoundaryType boundary_type_pressure = data.bc->pressure.get_boundary_type(boundary_id);
      BoundaryType boundary_type_energy   = data.bc->energy.get_boundary_type(boundary_id);

      EnergyBoundaryVariable boundary_variable = data.bc->energy.get_boundary_variable(boundary_id);

      for(unsigned int q = 0; q < density.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> flux = get_flux_boundary(density,
                                                                    momentum,
                                                                    energy,
                                                                    boundary_type_density,
                                                                    boundary_type_velocity,
                                                                    boundary_type_pressure,
                                                                    boundary_type_energy,
                                                                    boundary_variable,
                                                                    boundary_id,
                                                                    q);

        density.submit_value(std::get<0>(flux), q);
        momentum.submit_value(std::get<1>(flux), q);
        energy.submit_value(std::get<2>(flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::values, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  ConvectiveOperatorData<dim> data;

  // heat capacity ratio
  Number gamma;

  // specific gas constant
  Number R;

  // specific heat at constant volume
  Number c_v;

  mutable Number eval_time;
};


template<int dim>
struct ViscousOperatorData
{
  ViscousOperatorData()
    : dof_index(0),
      quad_index(0),
      IP_factor(1.0),
      dynamic_viscosity(1.0),
      reference_density(1.0),
      thermal_conductivity(0.0262),
      heat_capacity_ratio(1.4),
      specific_gas_constant(287.058)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  double IP_factor;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;

  double dynamic_viscosity;
  double reference_density;
  double thermal_conductivity;
  double heat_capacity_ratio;
  double specific_gas_constant;
};

template<int dim, typename Number>
class ViscousOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ViscousOperator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorVector;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;
  typedef dealii::Point<dim, dealii::VectorizedArray<Number>>     point;

  ViscousOperator() : matrix_free(nullptr), degree(1)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             ViscousOperatorData<dim> const &        data_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free->get_dof_handler(data.dof_index).get_fe();
    degree                                = fe.degree;

    gamma  = data.heat_capacity_ratio;
    R      = data.specific_gas_constant;
    c_v    = R / (gamma - 1.0);
    mu     = data.dynamic_viscosity;
    nu     = mu / data.reference_density;
    lambda = data.thermal_conductivity;

    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                                 *matrix_free,
                                                 data.dof_index);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

  void
  set_evaluation_time(double const & evaluation_time) const
  {
    eval_time = evaluation_time;
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_penalty_parameter(FaceIntegratorScalar & fe_eval_m, FaceIntegratorScalar & fe_eval_p) const
  {
    scalar tau =
      std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
               fe_eval_p.read_cell_data(array_penalty_parameter)) *
      IP::get_penalty_factor<dim, Number>(
        degree,
        GridUtilities::get_element_type(
          fe_eval_m.get_matrix_free().get_dof_handler(data.dof_index).get_triangulation()),
        data.IP_factor) *
      nu;

    return tau;
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_penalty_parameter(FaceIntegratorScalar & fe_eval) const
  {
    scalar tau = fe_eval.read_cell_data(array_penalty_parameter) *
                 IP::get_penalty_factor<dim, Number>(
                   degree,
                   GridUtilities::get_element_type(
                     fe_eval.get_matrix_free().get_dof_handler(data.dof_index).get_triangulation()),
                   data.IP_factor) *
                 nu;

    return tau;
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector, tensor, vector>
    get_volume_flux(CellIntegratorScalar & density,
                    CellIntegratorVector & momentum,
                    CellIntegratorScalar & energy,
                    unsigned int const     q) const
  {
    scalar rho_inv  = 1.0 / density.get_value(q);
    vector grad_rho = density.get_gradient(q);

    vector rho_u      = momentum.get_value(q);
    vector u          = rho_inv * rho_u;
    tensor grad_rho_u = momentum.get_gradient(q);

    scalar rho_E      = energy.get_value(q);
    vector grad_rho_E = energy.get_gradient(q);

    // calculate flux momentum
    tensor grad_u = calculate_grad_u(rho_inv, rho_u, grad_rho, grad_rho_u);
    tensor tau    = calculate_stress_tensor(grad_u, mu);

    // calculate flux energy
    vector grad_E      = calculate_grad_E(rho_inv, rho_E, grad_rho, grad_rho_E);
    vector grad_T      = calculate_grad_T(grad_E, u, grad_u, gamma, R);
    vector energy_flux = tau * u + lambda * grad_T;

    return std::make_tuple(vector() /* dummy */, tau, energy_flux);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, vector, scalar>
    get_gradient_flux(FaceIntegratorScalar & density_m,
                      FaceIntegratorScalar & density_p,
                      FaceIntegratorVector & momentum_m,
                      FaceIntegratorVector & momentum_p,
                      FaceIntegratorScalar & energy_m,
                      FaceIntegratorScalar & energy_p,
                      scalar const &         tau_IP,
                      unsigned int const     q) const
  {
    vector normal = momentum_m.get_normal_vector(q);

    // density
    scalar rho_M      = density_m.get_value(q);
    vector grad_rho_M = density_m.get_gradient(q);

    scalar rho_P      = density_p.get_value(q);
    vector grad_rho_P = density_p.get_gradient(q);

    // velocity
    vector rho_u_M      = momentum_m.get_value(q);
    tensor grad_rho_u_M = momentum_m.get_gradient(q);

    vector rho_u_P      = momentum_p.get_value(q);
    tensor grad_rho_u_P = momentum_p.get_gradient(q);

    // energy
    scalar rho_E_M      = energy_m.get_value(q);
    vector grad_rho_E_M = energy_m.get_gradient(q);

    scalar rho_E_P      = energy_p.get_value(q);
    vector grad_rho_E_P = energy_p.get_gradient(q);

    // flux density
    scalar jump_density          = rho_M - rho_P;
    scalar gradient_flux_density = -tau_IP * jump_density;

    // flux momentum
    scalar rho_inv_M = 1.0 / rho_M;
    tensor grad_u_M  = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
    tensor tau_M     = calculate_stress_tensor(grad_u_M, mu);

    scalar rho_inv_P = 1.0 / rho_P;
    tensor grad_u_P  = calculate_grad_u(rho_inv_P, rho_u_P, grad_rho_P, grad_rho_u_P);
    tensor tau_P     = calculate_stress_tensor(grad_u_P, mu);

    vector jump_momentum          = rho_u_M - rho_u_P;
    vector gradient_flux_momentum = 0.5 * (tau_M + tau_P) * normal - tau_IP * jump_momentum;

    // flux energy
    vector u_M      = rho_inv_M * rho_u_M;
    vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
    vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

    vector u_P      = rho_inv_P * rho_u_P;
    vector grad_E_P = calculate_grad_E(rho_inv_P, rho_E_P, grad_rho_P, grad_rho_E_P);
    vector grad_T_P = calculate_grad_T(grad_E_P, u_P, grad_u_P, gamma, R);

    vector flux_energy_average = 0.5 * (tau_M * u_M + tau_P * u_P + lambda * (grad_T_M + grad_T_P));

    scalar jump_energy          = rho_E_M - rho_E_P;
    scalar gradient_flux_energy = flux_energy_average * normal - tau_IP * jump_energy;

    return std::make_tuple(gradient_flux_density, gradient_flux_momentum, gradient_flux_energy);
  }


  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<scalar, vector, scalar>
    get_gradient_flux_boundary(FaceIntegratorScalar &             density,
                               FaceIntegratorVector &             momentum,
                               FaceIntegratorScalar &             energy,
                               scalar const &                     tau_IP,
                               BoundaryType const &               boundary_type_density,
                               BoundaryType const &               boundary_type_velocity,
                               BoundaryType const &               boundary_type_energy,
                               EnergyBoundaryVariable const &     boundary_variable,
                               dealii::types::boundary_id const & boundary_id,
                               unsigned int const                 q) const
  {
    vector normal = momentum.get_normal_vector(q);

    // density
    scalar rho_M      = density.get_value(q);
    vector grad_rho_M = density.get_gradient(q);

    scalar rho_P = calculate_exterior_value<dim, Number, 0>(rho_M,
                                                            boundary_type_density,
                                                            data.bc->density,
                                                            boundary_id,
                                                            density.quadrature_point(q),
                                                            this->eval_time);

    scalar jump_density          = rho_M - rho_P;
    scalar gradient_flux_density = -tau_IP * jump_density;

    // velocity
    vector rho_u_M      = momentum.get_value(q);
    tensor grad_rho_u_M = momentum.get_gradient(q);

    scalar rho_inv_M = 1.0 / rho_M;
    vector u_M       = rho_inv_M * rho_u_M;

    vector u_P = calculate_exterior_value<dim, Number, 1>(u_M,
                                                          boundary_type_velocity,
                                                          data.bc->velocity,
                                                          boundary_id,
                                                          momentum.quadrature_point(q),
                                                          this->eval_time);

    vector rho_u_P = rho_P * u_P;

    tensor grad_u_M = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
    tensor tau_M    = calculate_stress_tensor(grad_u_M, mu);

    vector tau_P_normal =
      calculate_exterior_normal_grad<dim, Number, 1>(tau_M * normal,
                                                     boundary_type_velocity,
                                                     data.bc->velocity,
                                                     boundary_id,
                                                     momentum.quadrature_point(q),
                                                     this->eval_time);

    vector jump_momentum          = rho_u_M - rho_u_P;
    vector gradient_flux_momentum = 0.5 * (tau_M * normal + tau_P_normal) - tau_IP * jump_momentum;

    // energy
    scalar rho_E_M      = energy.get_value(q);
    vector grad_rho_E_M = energy.get_gradient(q);

    scalar E_M = rho_inv_M * rho_E_M;
    scalar E_P = dealii::make_vectorized_array<Number>(0.0);
    if(boundary_variable == EnergyBoundaryVariable::Energy)
    {
      E_P = calculate_exterior_value<dim, Number, 0>(E_M,
                                                     boundary_type_energy,
                                                     data.bc->energy,
                                                     boundary_id,
                                                     energy.quadrature_point(q),
                                                     this->eval_time);
    }
    else if(boundary_variable == EnergyBoundaryVariable::Temperature)
    {
      scalar p_M = calculate_pressure(rho_M, u_M, E_M, gamma);
      scalar T_M = calculate_temperature(p_M, rho_M, R);
      scalar T_P = calculate_exterior_value<dim, Number, 0>(T_M,
                                                            boundary_type_energy,
                                                            data.bc->energy,
                                                            boundary_id,
                                                            energy.quadrature_point(q),
                                                            this->eval_time);

      E_P = calculate_energy(T_P, u_P, c_v);
    }

    scalar rho_E_P = rho_P * E_P;

    vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
    vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

    scalar grad_T_M_normal = grad_T_M * normal;
    scalar grad_T_P_normal =
      calculate_exterior_normal_grad<dim, Number, 0>(grad_T_M_normal,
                                                     boundary_type_energy,
                                                     data.bc->energy,
                                                     boundary_id,
                                                     energy.quadrature_point(q),
                                                     this->eval_time);

    scalar jump_energy          = rho_E_M - rho_E_P;
    scalar gradient_flux_energy = 0.5 * (u_M * tau_M * normal + u_P * tau_P_normal +
                                         lambda * (grad_T_M * normal + grad_T_P_normal)) -
                                  tau_IP * jump_energy;

    return std::make_tuple(gradient_flux_density, gradient_flux_momentum, gradient_flux_energy);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector /*dummy_M*/,
               tensor /*value_flux_momentum_M*/,
               vector /*value_flux_energy_M*/,
               vector /*dummy_P*/,
               tensor /*value_flux_momentum_P*/,
               vector /*value_flux_energy_P*/>
    get_value_flux(FaceIntegratorScalar & density_m,
                   FaceIntegratorScalar & density_p,
                   FaceIntegratorVector & momentum_m,
                   FaceIntegratorVector & momentum_p,
                   FaceIntegratorScalar & energy_m,
                   FaceIntegratorScalar & energy_p,
                   unsigned int const     q) const
  {
    vector normal = momentum_m.get_normal_vector(q);

    // density
    scalar rho_M = density_m.get_value(q);
    scalar rho_P = density_p.get_value(q);

    // velocity
    vector rho_u_M = momentum_m.get_value(q);
    vector rho_u_P = momentum_p.get_value(q);

    // energy
    scalar rho_E_M = energy_m.get_value(q);
    scalar rho_E_P = energy_p.get_value(q);

    vector jump_rho   = (rho_M - rho_P) * normal;
    tensor jump_rho_u = outer_product(rho_u_M - rho_u_P, normal);
    vector jump_rho_E = (rho_E_M - rho_E_P) * normal;

    scalar rho_inv_M = 1.0 / rho_M;
    scalar rho_inv_P = 1.0 / rho_P;

    vector u_M = rho_inv_M * rho_u_M;
    vector u_P = rho_inv_P * rho_u_P;

    // value flux momentum
    tensor grad_u_using_jumps_M = calculate_grad_u(rho_inv_M,
                                                   rho_u_M,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_u /*instead of grad_rho_u*/);

    tensor tau_using_jumps_M     = calculate_stress_tensor(grad_u_using_jumps_M, mu);
    tensor value_flux_momentum_M = -0.5 * tau_using_jumps_M;

    tensor grad_u_using_jumps_P = calculate_grad_u(rho_inv_P,
                                                   rho_u_P,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_u /*instead of grad_rho_u*/);

    tensor tau_using_jumps_P     = calculate_stress_tensor(grad_u_using_jumps_P, mu);
    tensor value_flux_momentum_P = -0.5 * tau_using_jumps_P;

    // value flux energy
    vector grad_E_using_jumps_M = calculate_grad_E(rho_inv_M,
                                                   rho_E_M,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_E /*instead of grad_rho_E*/);

    vector grad_T_using_jumps_M =
      calculate_grad_T(grad_E_using_jumps_M, u_M, grad_u_using_jumps_M, gamma, R);
    vector value_flux_energy_M = -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

    vector grad_E_using_jumps_P = calculate_grad_E(rho_inv_P,
                                                   rho_E_P,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_E /*instead of grad_rho_E*/);

    vector grad_T_using_jumps_P =
      calculate_grad_T(grad_E_using_jumps_P, u_P, grad_u_using_jumps_P, gamma, R);
    vector value_flux_energy_P = -0.5 * (tau_using_jumps_P * u_P + lambda * grad_T_using_jumps_P);

    return std::make_tuple(vector() /*dummy*/,
                           value_flux_momentum_M,
                           value_flux_energy_M,
                           vector() /*dummy*/,
                           value_flux_momentum_P,
                           value_flux_energy_P);
  }

  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<vector /*dummy_M*/, tensor /*value_flux_momentum_M*/, vector /*value_flux_energy_M*/>
    get_value_flux_boundary(FaceIntegratorScalar &             density,
                            FaceIntegratorVector &             momentum,
                            FaceIntegratorScalar &             energy,
                            BoundaryType const &               boundary_type_density,
                            BoundaryType const &               boundary_type_velocity,
                            BoundaryType const &               boundary_type_energy,
                            EnergyBoundaryVariable const &     boundary_variable,
                            dealii::types::boundary_id const & boundary_id,
                            unsigned int const                 q) const
  {
    vector normal = momentum.get_normal_vector(q);

    // density
    scalar rho_M = density.get_value(q);
    scalar rho_P = calculate_exterior_value<dim, Number, 0>(rho_M,
                                                            boundary_type_density,
                                                            data.bc->density,
                                                            boundary_id,
                                                            density.quadrature_point(q),
                                                            this->eval_time);

    scalar rho_inv_M = 1.0 / rho_M;

    // velocity
    vector rho_u_M = momentum.get_value(q);
    vector u_M     = rho_inv_M * rho_u_M;

    vector u_P = calculate_exterior_value<dim, Number, 1>(u_M,
                                                          boundary_type_velocity,
                                                          data.bc->velocity,
                                                          boundary_id,
                                                          momentum.quadrature_point(q),
                                                          this->eval_time);

    vector rho_u_P = rho_P * u_P;

    // energy
    scalar rho_E_M = energy.get_value(q);
    scalar E_M     = rho_inv_M * rho_E_M;

    scalar E_P = dealii::make_vectorized_array<Number>(0.0);
    if(boundary_variable == EnergyBoundaryVariable::Energy)
    {
      E_P = calculate_exterior_value<dim, Number, 0>(E_M,
                                                     boundary_type_energy,
                                                     data.bc->energy,
                                                     boundary_id,
                                                     energy.quadrature_point(q),
                                                     this->eval_time);
    }
    else if(boundary_variable == EnergyBoundaryVariable::Temperature)
    {
      scalar p_M = calculate_pressure(rho_M, u_M, E_M, gamma);
      scalar T_M = calculate_temperature(p_M, rho_M, R);
      scalar T_P = calculate_exterior_value<dim, Number, 0>(T_M,
                                                            boundary_type_energy,
                                                            data.bc->energy,
                                                            boundary_id,
                                                            energy.quadrature_point(q),
                                                            this->eval_time);

      E_P = calculate_energy(T_P, u_P, c_v);
    }
    scalar rho_E_P = rho_P * E_P;

    vector jump_rho   = (rho_M - rho_P) * normal;
    tensor jump_rho_u = outer_product(rho_u_M - rho_u_P, normal);
    vector jump_rho_E = (rho_E_M - rho_E_P) * normal;

    // value flux momentum
    tensor grad_u_using_jumps_M = calculate_grad_u(rho_inv_M,
                                                   rho_u_M,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_u /*instead of grad_rho_u*/);

    tensor tau_using_jumps_M     = calculate_stress_tensor(grad_u_using_jumps_M, mu);
    tensor value_flux_momentum_M = -0.5 * tau_using_jumps_M;

    // value flux energy
    vector grad_E_using_jumps_M = calculate_grad_E(rho_inv_M,
                                                   rho_E_M,
                                                   jump_rho /*instead of grad_rho*/,
                                                   jump_rho_E /*instead of grad_rho_E*/);

    vector grad_T_using_jumps_M =
      calculate_grad_T(grad_E_using_jumps_M, u_M, grad_u_using_jumps_M, gamma, R);
    vector value_flux_energy_M = -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

    return std::make_tuple(vector() /*dummy*/, value_flux_momentum_M, value_flux_energy_M);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar density(matrix_free, data.dof_index, data.quad_index, 0);
    CellIntegratorVector momentum(matrix_free, data.dof_index, data.quad_index, 1);
    CellIntegratorScalar energy(matrix_free, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      density.reinit(cell);
      density.gather_evaluate(src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      momentum.reinit(cell);
      momentum.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy.reinit(cell);
      energy.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for(unsigned int q = 0; q < momentum.n_q_points; ++q)
      {
        std::tuple<vector, tensor, vector> flux = get_volume_flux(density, momentum, energy, q);

        momentum.submit_gradient(std::get<1>(flux), q);
        energy.submit_gradient(std::get<2>(flux), q);
      }

      momentum.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }
  }

  void
  face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density_m(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorScalar density_p(matrix_free, false, data.dof_index, data.quad_index, 0);

    FaceIntegratorVector momentum_m(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorVector momentum_p(matrix_free, false, data.dof_index, data.quad_index, 1);

    FaceIntegratorScalar energy_m(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);
    FaceIntegratorScalar energy_p(matrix_free, false, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      density_m.reinit(face);
      density_m.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

      density_p.reinit(face);
      density_p.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

      // momentum
      momentum_m.reinit(face);
      momentum_m.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

      momentum_p.reinit(face);
      momentum_p.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

      // energy
      energy_m.reinit(face);
      energy_m.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy_p.reinit(face);
      energy_p.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      scalar tau_IP = get_penalty_parameter(density_m, density_p);

      for(unsigned int q = 0; q < density_m.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> gradient_flux = get_gradient_flux(
          density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, tau_IP, q);

        std::tuple<vector, tensor, vector, vector, tensor, vector> value_flux =
          get_value_flux(density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, q);

        density_m.submit_value(-std::get<0>(gradient_flux), q);
        // + sign since n⁺ = -n⁻
        density_p.submit_value(std::get<0>(gradient_flux), q);

        momentum_m.submit_gradient(std::get<1>(value_flux), q);
        // note that value_flux_momentum is not conservative
        momentum_p.submit_gradient(std::get<4>(value_flux), q);

        momentum_m.submit_value(-std::get<1>(gradient_flux), q);
        // + sign since n⁺ = -n⁻
        momentum_p.submit_value(std::get<1>(gradient_flux), q);

        energy_m.submit_gradient(std::get<2>(value_flux), q);
        // note that value_flux_energy is not conservative
        energy_p.submit_gradient(std::get<5>(value_flux), q);

        energy_m.submit_value(-std::get<2>(gradient_flux), q);
        // + sign since n⁺ = -n⁻
        energy_p.submit_value(std::get<2>(gradient_flux), q);
      }

      density_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      density_p.integrate_scatter(dealii::EvaluationFlags::values, dst);

      momentum_m.integrate_scatter(dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients,
                                   dst);
      momentum_p.integrate_scatter(dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients,
                                   dst);

      energy_m.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
      energy_p.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
    }
  }

  void
  boundary_face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
                     VectorType &                                  dst,
                     VectorType const &                            src,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorVector momentum(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorScalar energy(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      density.reinit(face);
      density.gather_evaluate(src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      momentum.reinit(face);
      momentum.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy.reinit(face);
      energy.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      scalar tau_IP = get_penalty_parameter(density);

      dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      BoundaryType boundary_type_density  = data.bc->density.get_boundary_type(boundary_id);
      BoundaryType boundary_type_velocity = data.bc->velocity.get_boundary_type(boundary_id);
      BoundaryType boundary_type_energy   = data.bc->energy.get_boundary_type(boundary_id);

      EnergyBoundaryVariable boundary_variable = data.bc->energy.get_boundary_variable(boundary_id);

      for(unsigned int q = 0; q < density.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> gradient_flux =
          get_gradient_flux_boundary(density,
                                     momentum,
                                     energy,
                                     tau_IP,
                                     boundary_type_density,
                                     boundary_type_velocity,
                                     boundary_type_energy,
                                     boundary_variable,
                                     boundary_id,
                                     q);

        std::tuple<vector, tensor, vector> value_flux =
          get_value_flux_boundary(density,
                                  momentum,
                                  energy,
                                  boundary_type_density,
                                  boundary_type_velocity,
                                  boundary_type_energy,
                                  boundary_variable,
                                  boundary_id,
                                  q);

        density.submit_value(-std::get<0>(gradient_flux), q);

        momentum.submit_gradient(std::get<1>(value_flux), q);
        momentum.submit_value(-std::get<1>(gradient_flux), q);

        energy.submit_gradient(std::get<2>(value_flux), q);
        energy.submit_value(-std::get<2>(gradient_flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
      energy.integrate_scatter(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                               dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  ViscousOperatorData<dim> data;

  unsigned int degree;

  // heat capacity ratio
  Number gamma;

  // specific gas constant
  Number R;

  // specific heat at constant volume
  Number c_v;

  // dynamic viscosity
  Number mu;

  // kinematic viscosity
  Number nu;

  // thermal conductivity
  Number lambda;

  dealii::AlignedVector<dealii::VectorizedArray<Number>> array_penalty_parameter;

  mutable Number eval_time;
};

template<int dim>
struct CombinedOperatorData
{
  CombinedOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};

template<int dim, typename Number>
class CombinedOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ConvectiveOperator<dim, Number> ConvectiveOp;
  typedef ViscousOperator<dim, Number>    ViscousOp;
  typedef CombinedOperator<dim, Number>   This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorVector;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;
  typedef dealii::Point<dim, dealii::VectorizedArray<Number>>     point;

  CombinedOperator() : matrix_free(nullptr), convective_operator(nullptr), viscous_operator(nullptr)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             CombinedOperatorData<dim> const &       data_in,
             ConvectiveOp const &                    convective_operator_in,
             ViscousOp const &                       viscous_operator_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;

    this->convective_operator = &convective_operator_in;
    this->viscous_operator    = &viscous_operator_in;
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    convective_operator->set_evaluation_time(evaluation_time);
    viscous_operator->set_evaluation_time(evaluation_time);

    matrix_free->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);

    // perform cell integrals only for performance measurements
    //    matrix_free->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar density(matrix_free, data.dof_index, data.quad_index, 0);
    CellIntegratorVector momentum(matrix_free, data.dof_index, data.quad_index, 1);
    CellIntegratorScalar energy(matrix_free, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      density.reinit(cell);
      density.gather_evaluate(src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      momentum.reinit(cell);
      momentum.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy.reinit(cell);
      energy.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for(unsigned int q = 0; q < momentum.n_q_points; ++q)
      {
        std::tuple<vector, tensor, vector> conv_flux =
          convective_operator->get_volume_flux(density, momentum, energy, q);

        std::tuple<vector, tensor, vector> visc_flux =
          viscous_operator->get_volume_flux(density, momentum, energy, q);

        density.submit_gradient(-std::get<0>(conv_flux), q);
        momentum.submit_gradient(-std::get<1>(conv_flux) + std::get<1>(visc_flux), q);
        energy.submit_gradient(-std::get<2>(conv_flux) + std::get<2>(visc_flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
      energy.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }
  }

  void
  face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density_m(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorScalar density_p(matrix_free, false, data.dof_index, data.quad_index, 0);
    FaceIntegratorVector momentum_m(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorVector momentum_p(matrix_free, false, data.dof_index, data.quad_index, 1);
    FaceIntegratorScalar energy_m(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);
    FaceIntegratorScalar energy_p(matrix_free, false, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      density_m.reinit(face);
      density_m.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

      density_p.reinit(face);
      density_p.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

      // momentum
      momentum_m.reinit(face);
      momentum_m.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

      momentum_p.reinit(face);
      momentum_p.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

      // energy
      energy_m.reinit(face);
      energy_m.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy_p.reinit(face);
      energy_p.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      scalar tau_IP = viscous_operator->get_penalty_parameter(density_m, density_p);

      for(unsigned int q = 0; q < density_m.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> conv_flux = convective_operator->get_flux(
          density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, q);

        std::tuple<scalar, vector, scalar> visc_grad_flux = viscous_operator->get_gradient_flux(
          density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, tau_IP, q);

        std::tuple<vector, tensor, vector, vector, tensor, vector> visc_value_flux =
          viscous_operator->get_value_flux(
            density_m, density_p, momentum_m, momentum_p, energy_m, energy_p, q);

        density_m.submit_value(std::get<0>(conv_flux) - std::get<0>(visc_grad_flux), q);
        // - sign since n⁺ = -n⁻
        density_p.submit_value(-std::get<0>(conv_flux) + std::get<0>(visc_grad_flux), q);

        momentum_m.submit_value(std::get<1>(conv_flux) - std::get<1>(visc_grad_flux), q);
        // - sign since n⁺ = -n⁻
        momentum_p.submit_value(-std::get<1>(conv_flux) + std::get<1>(visc_grad_flux), q);

        momentum_m.submit_gradient(std::get<1>(visc_value_flux), q);
        // note that value_flux_momentum is not conservative
        momentum_p.submit_gradient(std::get<4>(visc_value_flux), q);

        energy_m.submit_value(std::get<2>(conv_flux) - std::get<2>(visc_grad_flux), q);
        // - sign since n⁺ = -n⁻
        energy_p.submit_value(-std::get<2>(conv_flux) + std::get<2>(visc_grad_flux), q);

        energy_m.submit_gradient(std::get<2>(visc_value_flux), q);
        // note that value_flux_energy is not conservative
        energy_p.submit_gradient(std::get<5>(visc_value_flux), q);
      }

      density_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      density_p.integrate_scatter(dealii::EvaluationFlags::values, dst);

      momentum_m.integrate_scatter(dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients,
                                   dst);
      momentum_p.integrate_scatter(dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients,
                                   dst);

      energy_m.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
      energy_p.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
    }
  }

  void
  boundary_face_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
                     VectorType &                                  dst,
                     VectorType const &                            src,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceIntegratorScalar density(matrix_free, true, data.dof_index, data.quad_index, 0);
    FaceIntegratorVector momentum(matrix_free, true, data.dof_index, data.quad_index, 1);
    FaceIntegratorScalar energy(matrix_free, true, data.dof_index, data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      density.reinit(face);
      density.gather_evaluate(src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      momentum.reinit(face);
      momentum.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

      energy.reinit(face);
      energy.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      scalar tau_IP = viscous_operator->get_penalty_parameter(density);

      dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      BoundaryType boundary_type_density  = data.bc->density.get_boundary_type(boundary_id);
      BoundaryType boundary_type_velocity = data.bc->velocity.get_boundary_type(boundary_id);
      BoundaryType boundary_type_pressure = data.bc->pressure.get_boundary_type(boundary_id);
      BoundaryType boundary_type_energy   = data.bc->energy.get_boundary_type(boundary_id);

      EnergyBoundaryVariable boundary_variable = data.bc->energy.get_boundary_variable(boundary_id);

      for(unsigned int q = 0; q < density.n_q_points; ++q)
      {
        std::tuple<scalar, vector, scalar> conv_flux =
          convective_operator->get_flux_boundary(density,
                                                 momentum,
                                                 energy,
                                                 boundary_type_density,
                                                 boundary_type_velocity,
                                                 boundary_type_pressure,
                                                 boundary_type_energy,
                                                 boundary_variable,
                                                 boundary_id,
                                                 q);

        std::tuple<scalar, vector, scalar> visc_grad_flux =
          viscous_operator->get_gradient_flux_boundary(density,
                                                       momentum,
                                                       energy,
                                                       tau_IP,
                                                       boundary_type_density,
                                                       boundary_type_velocity,
                                                       boundary_type_energy,
                                                       boundary_variable,
                                                       boundary_id,
                                                       q);

        std::tuple<vector, tensor, vector> visc_value_flux =
          viscous_operator->get_value_flux_boundary(density,
                                                    momentum,
                                                    energy,
                                                    boundary_type_density,
                                                    boundary_type_velocity,
                                                    boundary_type_energy,
                                                    boundary_variable,
                                                    boundary_id,
                                                    q);

        density.submit_value(std::get<0>(conv_flux) - std::get<0>(visc_grad_flux), q);

        momentum.submit_value(std::get<1>(conv_flux) - std::get<1>(visc_grad_flux), q);
        momentum.submit_gradient(std::get<1>(visc_value_flux), q);

        energy.submit_value(std::get<2>(conv_flux) - std::get<2>(visc_grad_flux), q);
        energy.submit_gradient(std::get<2>(visc_value_flux), q);
      }

      density.integrate_scatter(dealii::EvaluationFlags::values, dst);
      momentum.integrate_scatter(dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients,
                                 dst);
      energy.integrate_scatter(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                               dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  CombinedOperatorData<dim> data;

  ConvectiveOperator<dim, Number> const * convective_operator;
  ViscousOperator<dim, Number> const *    viscous_operator;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_KERNELS_AND_OPERATORS_H_ \
        */
