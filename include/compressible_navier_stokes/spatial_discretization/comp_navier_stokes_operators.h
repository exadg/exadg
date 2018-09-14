/* *CompNavierStokesOperators.h
 *
 *
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_OPERATORS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_OPERATORS_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "../include/functionalities/evaluate_functions.h"
#include "operators/interior_penalty_parameter.h"

#include "../../compressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../compressible_navier_stokes/user_interface/input_parameters.h"


#include <iostream>


namespace CompNS
{
enum class BoundaryType
{
  undefined,
  dirichlet,
  neumann
};

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_pressure(Tensor<1, dim, VectorizedArray<value_type>> const & rho_u,
                       Tensor<1, dim, VectorizedArray<value_type>> const & u,
                       VectorizedArray<value_type> const &                 rho_E,
                       value_type const &                                  gamma)
{
  return (gamma - 1.0) * (rho_E - 0.5 * scalar_product(rho_u, u));
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  calculate_pressure(VectorizedArray<value_type> const &                 rho,
                     Tensor<1, dim, VectorizedArray<value_type>> const & u,
                     VectorizedArray<value_type> const &                 E,
                     value_type const &                                  gamma)
{
  return (gamma - 1.0) * rho * (E - 0.5 * scalar_product(u, u));
}

template<typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  calculate_temperature(VectorizedArray<value_type> const & p,
                        VectorizedArray<value_type> const & rho,
                        value_type const &                  R)
{
  return p / (rho * R);
}

template<int dim, typename value_type>
inline VectorizedArray<value_type>
calculate_energy(VectorizedArray<value_type> const &                 T,
                 Tensor<1, dim, VectorizedArray<value_type>> const & u,
                 value_type const &                                  c_v)
{
  return c_v * T + 0.5 * scalar_product(u, u);
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<value_type>>
  calculate_grad_E(VectorizedArray<value_type> const &                 rho_inverse,
                   VectorizedArray<value_type> const &                 rho_E,
                   Tensor<1, dim, VectorizedArray<value_type>> const & grad_rho,
                   Tensor<1, dim, VectorizedArray<value_type>> const & grad_rho_E)
{
  VectorizedArray<value_type> E = rho_inverse * rho_E;

  return rho_inverse * (grad_rho_E - E * grad_rho);
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, dim, VectorizedArray<value_type>>
  calculate_grad_u(VectorizedArray<value_type> const &                 rho_inverse,
                   Tensor<1, dim, VectorizedArray<value_type>> const & rho_u,
                   Tensor<1, dim, VectorizedArray<value_type>> const & grad_rho,
                   Tensor<2, dim, VectorizedArray<value_type>> const & grad_rho_u)
{
  Tensor<2, dim, VectorizedArray<value_type>> out;
  for(unsigned int d = 0; d < dim; ++d)
  {
    VectorizedArray<value_type> ud = rho_inverse * rho_u[d];
    for(unsigned int e = 0; e < dim; ++e)
      out[d][e] = rho_inverse * (grad_rho_u[d][e] - ud * grad_rho[e]);
  }
  return out;
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    calculate_grad_T(Tensor<1, dim, VectorizedArray<value_type>> const & grad_E,
                     Tensor<1, dim, VectorizedArray<value_type>> const & u,
                     Tensor<2, dim, VectorizedArray<value_type>> const & grad_u,
                     value_type const &                                  gamma,
                     value_type const &                                  R)
{
  return (gamma - 1.0) / R * (grad_E - u * grad_u);
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<value_type>>
    calculate_stress_tensor(Tensor<2, dim, VectorizedArray<value_type>> const & grad_u,
                            value_type const &                                  viscosity)
{
  Tensor<2, dim, VectorizedArray<value_type>> out;
  const VectorizedArray<value_type>           divu = (2. / 3.) * trace(grad_u);
  for(unsigned int d = 0; d < dim; ++d)
  {
    for(unsigned int e = 0; e < dim; ++e)
      out[d][e] = viscosity * (grad_u[d][e] + grad_u[e][d]);
    out[d][d] -= viscosity * divu;
  }
  return out;
}

/*
 * Calculate exterior state "+" for a scalar quantity depending on interior state "-" and boundary
 * conditions.
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  calculate_exterior_value(
    VectorizedArray<value_type> const &                      value_m,
    BoundaryType const &                                     boundary_type,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>> const & boundary_descriptor,
    types::boundary_id const &                               boundary_id,
    Point<dim, VectorizedArray<value_type>> const &          q_point,
    value_type const &                                       time)
{
  VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.);

  if(boundary_type == BoundaryType::dirichlet)
  {
    VectorizedArray<value_type> g;

    typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
    it = boundary_descriptor->dirichlet_bc.find(boundary_id);

    evaluate_scalar_function(g, it->second, q_point, time);
    value_p = -value_m + make_vectorized_array<value_type>(2.0) * g;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return value_p;
}

/*
 * Calculate exterior state "+" for a scalar quantity depending on interior state "-" and boundary
 * conditions.
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    calculate_exterior_value(
      Tensor<1, dim, VectorizedArray<value_type>> const &      value_m,
      BoundaryType const                                       boundary_type,
      std::shared_ptr<CompNS::BoundaryDescriptor<dim>> const & boundary_descriptor,
      types::boundary_id const &                               boundary_id,
      Point<dim, VectorizedArray<value_type>> const &          q_point,
      value_type const &                                       time)
{
  Tensor<1, dim, VectorizedArray<value_type>> value_p;

  if(boundary_type == BoundaryType::dirichlet)
  {
    Tensor<1, dim, VectorizedArray<value_type>> g;

    typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
    it = boundary_descriptor->dirichlet_bc.find(boundary_id);

    evaluate_vectorial_function(g, it->second, q_point, time);
    value_p = -value_m + make_vectorized_array<value_type>(2.0) * g;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return value_p;
}

/*
 * Calculate exterior state of normal stresses depending on interior data and boundary conditions.
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    calculate_exterior_normal_grad(
      Tensor<1, dim, VectorizedArray<value_type>> const &      tau_M_normal,
      BoundaryType const &                                     boundary_type,
      std::shared_ptr<CompNS::BoundaryDescriptor<dim>> const & boundary_descriptor,
      types::boundary_id const &                               boundary_id,
      Point<dim, VectorizedArray<value_type>> const &          q_point,
      value_type const &                                       time)
{
  Tensor<1, dim, VectorizedArray<value_type>> tau_P_normal;

  if(boundary_type == BoundaryType::dirichlet)
  {
    // do nothing
    tau_P_normal = tau_M_normal;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    Tensor<1, dim, VectorizedArray<value_type>> h;

    typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
    it = boundary_descriptor->neumann_bc.find(boundary_id);
    evaluate_vectorial_function(h, it->second, q_point, time);

    tau_P_normal = -tau_M_normal + make_vectorized_array<value_type>(2.0) * h;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return tau_P_normal;
}

/*
 * Calculate exterior temperature gradient depending on interior data and boundary conditions.
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  calculate_exterior_normal_grad(
    VectorizedArray<value_type> const &                      grad_T_M_normal,
    BoundaryType const &                                     boundary_type,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>> const & boundary_descriptor,
    types::boundary_id const &                               boundary_id,
    Point<dim, VectorizedArray<value_type>> const &          q_point,
    value_type const &                                       time)
{
  VectorizedArray<value_type> grad_T_P_normal = make_vectorized_array<value_type>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    // do nothing
    grad_T_P_normal = grad_T_M_normal;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    VectorizedArray<value_type> h;

    typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
    it = boundary_descriptor->neumann_bc.find(boundary_id);
    evaluate_scalar_function(h, it->second, q_point, time);

    grad_T_P_normal = -grad_T_M_normal + make_vectorized_array<value_type>(2.0) * h;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return grad_T_P_normal;
}

/*
 * This function calculates the Lax-Friedrichs flux for the momentum equation
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    calculate_flux(Tensor<2, dim, VectorizedArray<value_type>> const & momentum_flux_M,
                   Tensor<2, dim, VectorizedArray<value_type>> const & momentum_flux_P,
                   Tensor<1, dim, VectorizedArray<value_type>> const & rho_u_M,
                   Tensor<1, dim, VectorizedArray<value_type>> const & rho_u_P,
                   VectorizedArray<value_type> const &                 lambda,
                   Tensor<1, dim, VectorizedArray<value_type>> const & normal)
{
  Tensor<1, dim, VectorizedArray<value_type>> out;
  for(unsigned int d = 0; d < dim; ++d)
  {
    VectorizedArray<value_type> sum = VectorizedArray<value_type>();
    for(unsigned int e = 0; e < dim; ++e)
      sum += (momentum_flux_M[d][e] + momentum_flux_P[d][e]) * normal[e];
    out[d] = 0.5 * (sum + lambda * (rho_u_M[d] - rho_u_P[d]));
  }
  return out;
}

/*
 * This function calculates the Lax-Friedrichs flux for scalar quantities (density/energy)
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_flux(Tensor<1, dim, VectorizedArray<value_type>> const & flux_M,
                   Tensor<1, dim, VectorizedArray<value_type>> const & flux_P,
                   VectorizedArray<value_type> const &                 value_M,
                   VectorizedArray<value_type> const &                 value_P,
                   VectorizedArray<value_type> const &                 lambda,
                   Tensor<1, dim, VectorizedArray<value_type>> const & normal)
{
  Tensor<1, dim, VectorizedArray<value_type>> average_flux = 0.5 * (flux_M + flux_P);

  return average_flux * normal + 0.5 * lambda * (value_M - value_P);
}

/*
 * Calculation of lambda for Lax-Friedrichs flux according to Hesthaven:
 *   lambda = max( |u_M| + sqrt(|gamma * p_M / rho_M|) , |u_P| + sqrt(|gamma * p_P / rho_P|) )
 */
template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  calculate_lambda(VectorizedArray<value_type> const &                 rho_m,
                   VectorizedArray<value_type> const &                 rho_p,
                   Tensor<1, dim, VectorizedArray<value_type>> const & u_m,
                   Tensor<1, dim, VectorizedArray<value_type>> const & u_p,
                   VectorizedArray<value_type> const &                 p_m,
                   VectorizedArray<value_type> const &                 p_p,
                   value_type const &                                  gamma)
{
  VectorizedArray<value_type> lambda_m = u_m.norm() + std::sqrt(std::abs(gamma * p_m / rho_m));
  VectorizedArray<value_type> lambda_p = u_p.norm() + std::sqrt(std::abs(gamma * p_p / rho_p));

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

  std::shared_ptr<Function<dim>> rhs_rho;
  std::shared_ptr<Function<dim>> rhs_u;
  std::shared_ptr<Function<dim>> rhs_E;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class BodyForceOperator
{
public:
  BodyForceOperator() : data(nullptr), eval_time(0.0)
  {
  }

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEEval_velocity;

  typedef BodyForceOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  typedef VectorizedArray<value_type>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<value_type>> vector;
  typedef Tensor<2, dim, VectorizedArray<value_type>> tensor;
  typedef Point<dim, VectorizedArray<value_type>>     point;

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             BodyForceOperatorData<dim> const &  operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  evaluate(parallel::distributed::Vector<value_type> &       dst,
           parallel::distributed::Vector<value_type> const & src,
           double const                                      evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(parallel::distributed::Vector<value_type> &       dst,
               parallel::distributed::Vector<value_type> const & src,
               double const                                      evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEval_scalar   fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_velocity fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.gather_evaluate(src, true, false);
      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.gather_evaluate(src, true, false);
      fe_eval_energy.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
      {
        point  q_points = fe_eval_density.quadrature_point(q);
        scalar density  = fe_eval_density.get_value(q);
        vector velocity = fe_eval_momentum.get_value(q) / density;

        scalar rhs_density, rhs_energy;
        vector rhs_momentum;

        evaluate_scalar_function(rhs_density, operator_data.rhs_rho, q_points, eval_time);
        evaluate_vectorial_function(rhs_momentum, operator_data.rhs_u, q_points, eval_time);
        evaluate_scalar_function(rhs_energy, operator_data.rhs_E, q_points, eval_time);

        fe_eval_density.submit_value(rhs_density, q);
        fe_eval_momentum.submit_value(rhs_momentum, q);
        fe_eval_energy.submit_value(rhs_momentum * velocity + rhs_energy, q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_momentum.integrate_scatter(true, false, dst);
      fe_eval_energy.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  BodyForceOperatorData<dim> operator_data;

  double mutable eval_time;
};

struct MassMatrixOperatorData
{
  MassMatrixOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  MassMatrixOperator() : data(nullptr)
  {
  }

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEEval_velocity;

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             MassMatrixOperatorData const &      mass_matrix_operator_data_in)
  {
    this->data                      = &mf_data;
    this->mass_matrix_operator_data = mass_matrix_operator_data_in;
  }

  // apply matrix vector multiplication
  void
  apply(parallel::distributed::Vector<value_type> &       dst,
        const parallel::distributed::Vector<value_type> & src) const
  {
    dst = 0;
    apply_add(dst, src);
  }

  void
  apply_add(parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src) const
  {
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEval_scalar fe_eval_density(data,
                                  mass_matrix_operator_data.dof_index,
                                  mass_matrix_operator_data.quad_index,
                                  0);

    FEEval_velocity fe_eval_momentum(data,
                                     mass_matrix_operator_data.dof_index,
                                     mass_matrix_operator_data.quad_index,
                                     1);

    FEEval_scalar fe_eval_energy(data,
                                 mass_matrix_operator_data.dof_index,
                                 mass_matrix_operator_data.quad_index,
                                 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.gather_evaluate(src, true, false);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.gather_evaluate(src, true, false);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
      {
        fe_eval_density.submit_value(fe_eval_density.get_value(q), q);
        fe_eval_momentum.submit_value(fe_eval_momentum.get_value(q), q);
        fe_eval_energy.submit_value(fe_eval_energy.get_value(q), q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_momentum.integrate_scatter(true, false, dst);
      fe_eval_energy.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, value_type> const * data;
  MassMatrixOperatorData              mass_matrix_operator_data;
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

  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_rho;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_u;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_p;
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> bc_E;

  double heat_capacity_ratio;
  double specific_gas_constant;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class ConvectiveOperator
{
public:
  typedef ConvectiveOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  ConvectiveOperator() : data(nullptr)
  {
  }

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>       FEEval_scalar;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEFaceEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type>     FEEval_vectorial;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEFaceEval_vectorial;

  typedef VectorizedArray<value_type>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<value_type>> vector;
  typedef Tensor<2, dim, VectorizedArray<value_type>> tensor;
  typedef Point<dim, VectorizedArray<value_type>>     point;

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             ConvectiveOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  evaluate(parallel::distributed::Vector<value_type> &       dst,
           parallel::distributed::Vector<value_type> const & src,
           value_type const                                  evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(parallel::distributed::Vector<value_type> &       dst,
               parallel::distributed::Vector<value_type> const & src,
               value_type const                                  evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEval_scalar    fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma = operator_data.heat_capacity_ratio;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.gather_evaluate(src, true, false);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.gather_evaluate(src, true, false);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        scalar rho_inv         = 1.0 / fe_eval_density.get_value(q);
        vector rho_u           = fe_eval_momentum.get_value(q);
        scalar rho_E           = fe_eval_energy.get_value(q);
        vector u               = rho_inv * rho_u;
        tensor convective_flux = outer_product(rho_u, u);
        scalar p               = calculate_pressure(rho_u, u, rho_E, gamma);
        for(unsigned int d = 0; d < dim; ++d)
          convective_flux[d][d] += p;

        fe_eval_density.submit_gradient(-rho_u, q);
        fe_eval_momentum.submit_gradient(-convective_flux, q);
        fe_eval_energy.submit_gradient(-(rho_E + p) * u, q);
      }

      fe_eval_density.integrate_scatter(false, true, dst);
      fe_eval_momentum.integrate_scatter(false, true, dst);
      fe_eval_energy.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_scalar fe_eval_density_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 0);

    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_vectorial fe_eval_momentum_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1);

    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);
    FEFaceEval_scalar fe_eval_energy_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma = operator_data.heat_capacity_ratio;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, false);

      fe_eval_density_neighbor.reinit(face);
      fe_eval_density_neighbor.gather_evaluate(src, true, false);

      // velocity
      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, false);

      fe_eval_momentum_neighbor.reinit(face);
      fe_eval_momentum_neighbor.gather_evaluate(src, true, false);

      // energy
      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, false);

      fe_eval_energy_neighbor.reinit(face);
      fe_eval_energy_neighbor.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // get values
        scalar rho_M   = fe_eval_density.get_value(q);
        scalar rho_P   = fe_eval_density_neighbor.get_value(q);
        vector rho_u_M = fe_eval_momentum.get_value(q);
        vector rho_u_P = fe_eval_momentum_neighbor.get_value(q);
        vector u_M     = rho_u_M / rho_M;
        vector u_P     = rho_u_P / rho_P;
        scalar rho_E_M = fe_eval_energy.get_value(q);
        scalar rho_E_P = fe_eval_energy_neighbor.get_value(q);

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

        fe_eval_density.submit_value(flux_density, q);
        fe_eval_density_neighbor.submit_value(-flux_density, q);

        fe_eval_momentum.submit_value(flux_momentum, q);
        fe_eval_momentum_neighbor.submit_value(-flux_momentum, q);

        fe_eval_energy.submit_value(flux_energy, q);
        fe_eval_energy_neighbor.submit_value(-flux_energy, q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_density_neighbor.integrate_scatter(true, false, dst);

      fe_eval_momentum.integrate_scatter(true, false, dst);
      fe_eval_momentum_neighbor.integrate_scatter(true, false, dst);

      fe_eval_energy.integrate_scatter(true, false, dst);
      fe_eval_energy_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop(const MatrixFree<dim, value_type> &               data,
                     parallel::distributed::Vector<value_type> &       dst,
                     const parallel::distributed::Vector<value_type> & src,
                     const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma = operator_data.heat_capacity_ratio;
    value_type const R     = operator_data.specific_gas_constant;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      // density
      BoundaryType boundary_type_density = BoundaryType::undefined;
      if(operator_data.bc_rho->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_rho->dirichlet_bc.end())
      {
        boundary_type_density = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_rho->neumann_bc.find(boundary_id) !=
              operator_data.bc_rho->neumann_bc.end())
      {
        boundary_type_density = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_density != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      // velocity
      BoundaryType boundary_type_velocity = BoundaryType::undefined;
      if(operator_data.bc_u->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_u->dirichlet_bc.end())
      {
        boundary_type_velocity = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_u->neumann_bc.find(boundary_id) !=
              operator_data.bc_u->neumann_bc.end())
      {
        boundary_type_velocity = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_velocity != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      // pressure
      BoundaryType boundary_type_pressure = BoundaryType::undefined;
      if(operator_data.bc_p->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_p->dirichlet_bc.end())
      {
        boundary_type_pressure = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_p->neumann_bc.find(boundary_id) !=
              operator_data.bc_p->neumann_bc.end())
      {
        boundary_type_pressure = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_pressure != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      // energy
      BoundaryType boundary_type_energy = BoundaryType::undefined;
      if(operator_data.bc_E->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_E->dirichlet_bc.end())
      {
        boundary_type_energy = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_E->neumann_bc.find(boundary_id) !=
              operator_data.bc_E->neumann_bc.end())
      {
        boundary_type_energy = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_energy != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      EnergyBoundaryVariable boundary_variable =
        operator_data.bc_E->boundary_variable.find(boundary_id)->second;
      AssertThrow(boundary_variable != EnergyBoundaryVariable::Undefined,
                  ExcMessage("Energy boundary variable is undefined!"));

      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, false);

      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, false);

      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // element e⁻
        scalar rho_M   = fe_eval_density.get_value(q);
        vector rho_u_M = fe_eval_momentum.get_value(q);
        vector u_M     = rho_u_M / rho_M;
        scalar rho_E_M = fe_eval_energy.get_value(q);
        scalar E_M     = rho_E_M / rho_M;
        scalar p_M     = calculate_pressure(rho_M, u_M, E_M, gamma);

        // element e⁺

        // calculate rho_P
        scalar rho_P =
          calculate_exterior_value<dim, value_type>(rho_M,
                                                    boundary_type_density,
                                                    operator_data.bc_rho,
                                                    boundary_id,
                                                    fe_eval_density.quadrature_point(q),
                                                    eval_time);

        // calculate u_P
        vector u_P = calculate_exterior_value<dim, value_type>(u_M,
                                                               boundary_type_velocity,
                                                               operator_data.bc_u,
                                                               boundary_id,
                                                               fe_eval_momentum.quadrature_point(q),
                                                               eval_time);

        vector rho_u_P = rho_P * u_P;

        // calculate p_P
        scalar p_P = calculate_exterior_value<dim, value_type>(p_M,
                                                               boundary_type_pressure,
                                                               operator_data.bc_p,
                                                               boundary_id,
                                                               fe_eval_density.quadrature_point(q),
                                                               eval_time);

        // calculate E_P
        scalar E_P = make_vectorized_array<value_type>(0.0);
        if(boundary_variable == EnergyBoundaryVariable::Energy)
        {
          E_P = calculate_exterior_value<dim, value_type>(E_M,
                                                          boundary_type_energy,
                                                          operator_data.bc_E,
                                                          boundary_id,
                                                          fe_eval_energy.quadrature_point(q),
                                                          eval_time);
        }
        else if(boundary_variable == EnergyBoundaryVariable::Temperature)
        {
          scalar T_M = calculate_temperature(p_M, rho_M, R);
          scalar T_P = calculate_exterior_value<dim, value_type>(T_M,
                                                                 boundary_type_energy,
                                                                 operator_data.bc_E,
                                                                 boundary_id,
                                                                 fe_eval_energy.quadrature_point(q),
                                                                 eval_time);

          value_type const c_v = R / (gamma - 1.0);

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

        fe_eval_density.submit_value(flux_density, q);
        fe_eval_momentum.submit_value(flux_momentum, q);
        fe_eval_energy.submit_value(flux_energy, q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_momentum.integrate_scatter(true, false, dst);
      fe_eval_energy.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  ConvectiveOperatorData<dim> operator_data;

  mutable value_type eval_time;
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

  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_rho;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_u;
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> bc_E;

  double dynamic_viscosity;
  double reference_density;
  double thermal_conductivity;
  double heat_capacity_ratio;
  double specific_gas_constant;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class ViscousOperator
{
public:
  typedef ViscousOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  ViscousOperator() : data(nullptr)
  {
  }

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>       FEEval_scalar;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEFaceEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type>     FEEval_vectorial;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEFaceEval_vectorial;

  typedef VectorizedArray<value_type>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<value_type>> vector;
  typedef Tensor<2, dim, VectorizedArray<value_type>> tensor;
  typedef Point<dim, VectorizedArray<value_type>>     point;

  void
  initialize(Mapping<dim> const &                mapping,
             MatrixFree<dim, value_type> const & mf_data,
             ViscousOperatorData<dim> const &    operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, fe_degree, value_type>(array_penalty_parameter,
                                                                *data,
                                                                mapping,
                                                                operator_data.dof_index);
  }

  void
  evaluate(parallel::distributed::Vector<value_type> &       dst,
           const parallel::distributed::Vector<value_type> & src,
           value_type const                                  evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(parallel::distributed::Vector<value_type> &       dst,
               const parallel::distributed::Vector<value_type> & src,
               value_type const                                  evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEval_scalar    fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.gather_evaluate(src, true, true);

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        scalar rho_inv  = 1.0 / fe_eval_density.get_value(q);
        vector grad_rho = fe_eval_density.get_gradient(q);

        vector rho_u      = fe_eval_momentum.get_value(q);
        vector u          = rho_inv * rho_u;
        tensor grad_rho_u = fe_eval_momentum.get_gradient(q);

        scalar rho_E      = fe_eval_energy.get_value(q);
        vector grad_rho_E = fe_eval_energy.get_gradient(q);

        // calculate flux momentum
        tensor grad_u = calculate_grad_u(rho_inv, rho_u, grad_rho, grad_rho_u);
        tensor tau    = calculate_stress_tensor(grad_u, mu);

        // calculate flux energy
        vector grad_E      = calculate_grad_E(rho_inv, rho_E, grad_rho, grad_rho_E);
        vector grad_T      = calculate_grad_T(grad_E, u, grad_u, gamma, R);
        vector flux_energy = tau * u + lambda * grad_T;

        fe_eval_momentum.submit_gradient(tau, q);
        fe_eval_energy.submit_gradient(flux_energy, q);
      }

      fe_eval_momentum.integrate_scatter(false, true, dst);
      fe_eval_energy.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_scalar fe_eval_density_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 0);

    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_vectorial fe_eval_momentum_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1);

    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);
    FEFaceEval_scalar fe_eval_energy_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const nu     = mu / operator_data.reference_density;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_density_neighbor.reinit(face);
      fe_eval_density_neighbor.gather_evaluate(src, true, true);

      // momentum
      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_momentum_neighbor.reinit(face);
      fe_eval_momentum_neighbor.gather_evaluate(src, true, true);

      // energy
      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, true);

      fe_eval_energy_neighbor.reinit(face);
      fe_eval_energy_neighbor.gather_evaluate(src, true, true);

      VectorizedArray<value_type> tau_IP =
        std::max(fe_eval_momentum.read_cell_data(array_penalty_parameter),
                 fe_eval_momentum_neighbor.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor) * nu;

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // density
        scalar rho_M      = fe_eval_density.get_value(q);
        scalar rho_inv_M  = 1.0 / rho_M;
        vector grad_rho_M = fe_eval_density.get_gradient(q);

        scalar rho_P      = fe_eval_density_neighbor.get_value(q);
        scalar rho_inv_P  = 1.0 / rho_P;
        vector grad_rho_P = fe_eval_density_neighbor.get_gradient(q);

        scalar jump_density          = rho_M - rho_P;
        scalar gradient_flux_density = -tau_IP * jump_density;

        // velocity
        vector rho_u_M      = fe_eval_momentum.get_value(q);
        vector u_M          = rho_inv_M * rho_u_M;
        tensor grad_rho_u_M = fe_eval_momentum.get_gradient(q);

        vector rho_u_P      = fe_eval_momentum_neighbor.get_value(q);
        vector u_P          = rho_inv_P * rho_u_P;
        tensor grad_rho_u_P = fe_eval_momentum_neighbor.get_gradient(q);

        tensor grad_u_M = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
        tensor tau_M    = calculate_stress_tensor(grad_u_M, mu);

        tensor grad_u_P = calculate_grad_u(rho_inv_P, rho_u_P, grad_rho_P, grad_rho_u_P);
        tensor tau_P    = calculate_stress_tensor(grad_u_P, mu);

        vector jump_momentum          = rho_u_M - rho_u_P;
        vector gradient_flux_momentum = 0.5 * (tau_M + tau_P) * normal - tau_IP * jump_momentum;

        // energy
        scalar rho_E_M      = fe_eval_energy.get_value(q);
        vector grad_rho_E_M = fe_eval_energy.get_gradient(q);
        scalar rho_E_P      = fe_eval_energy_neighbor.get_value(q);
        vector grad_rho_E_P = fe_eval_energy_neighbor.get_gradient(q);

        vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
        vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

        vector grad_E_P = calculate_grad_E(rho_inv_P, rho_E_P, grad_rho_P, grad_rho_E_P);
        vector grad_T_P = calculate_grad_T(grad_E_P, u_P, grad_u_P, gamma, R);

        vector flux_energy_average =
          0.5 * (tau_M * u_M + tau_P * u_P + lambda * (grad_T_M + grad_T_P));

        scalar jump_energy          = rho_E_M - rho_E_P;
        scalar gradient_flux_energy = flux_energy_average * normal - tau_IP * jump_energy;

        // value flux momentum
        vector jump_rho   = jump_density * normal;
        vector jump_rho_E = jump_energy * normal;
        tensor jump_rho_u = outer_product(jump_momentum, normal);

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
        vector value_flux_energy_M =
          -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

        vector grad_E_using_jumps_P = calculate_grad_E(rho_inv_P,
                                                       rho_E_P,
                                                       jump_rho /*instead of grad_rho*/,
                                                       jump_rho_E /*instead of grad_rho_E*/);

        vector grad_T_using_jumps_P =
          calculate_grad_T(grad_E_using_jumps_P, u_P, grad_u_using_jumps_P, gamma, R);
        vector value_flux_energy_P =
          -0.5 * (tau_using_jumps_P * u_P + lambda * grad_T_using_jumps_P);

        fe_eval_density.submit_value(-gradient_flux_density, q);
        fe_eval_density_neighbor.submit_value(gradient_flux_density, q); // + sign since n⁺ = -n⁻

        fe_eval_momentum.submit_gradient(value_flux_momentum_M, q);
        // note that value_flux_momentum is not conservative
        fe_eval_momentum_neighbor.submit_gradient(value_flux_momentum_P, q);

        fe_eval_momentum.submit_value(-gradient_flux_momentum, q);
        fe_eval_momentum_neighbor.submit_value(gradient_flux_momentum, q); // + sign since n⁺ = -n⁻

        fe_eval_energy.submit_gradient(value_flux_energy_M, q);
        // note that value_flux_energy is not conservative
        fe_eval_energy_neighbor.submit_gradient(value_flux_energy_P, q);

        fe_eval_energy.submit_value(-gradient_flux_energy, q);
        fe_eval_energy_neighbor.submit_value(gradient_flux_energy, q); // + sign since n⁺ = -n⁻
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_density_neighbor.integrate_scatter(true, false, dst);

      fe_eval_momentum.integrate_scatter(true, true, dst);
      fe_eval_momentum_neighbor.integrate_scatter(true, true, dst);

      fe_eval_energy.integrate_scatter(true, true, dst);
      fe_eval_energy_neighbor.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop(const MatrixFree<dim, value_type> &               data,
                     parallel::distributed::Vector<value_type> &       dst,
                     const parallel::distributed::Vector<value_type> & src,
                     const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const nu     = mu / operator_data.reference_density;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      BoundaryType boundary_type_density = BoundaryType::undefined;
      if(operator_data.bc_rho->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_rho->dirichlet_bc.end())
      {
        boundary_type_density = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_rho->neumann_bc.find(boundary_id) !=
              operator_data.bc_rho->neumann_bc.end())
      {
        boundary_type_density = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_density != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      BoundaryType boundary_type_velocity = BoundaryType::undefined;
      if(operator_data.bc_u->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_u->dirichlet_bc.end())
      {
        boundary_type_velocity = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_u->neumann_bc.find(boundary_id) !=
              operator_data.bc_u->neumann_bc.end())
      {
        boundary_type_velocity = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_velocity != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      BoundaryType boundary_type_energy = BoundaryType::undefined;
      if(operator_data.bc_E->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_E->dirichlet_bc.end())
      {
        boundary_type_energy = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_E->neumann_bc.find(boundary_id) !=
              operator_data.bc_E->neumann_bc.end())
      {
        boundary_type_energy = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_energy != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      EnergyBoundaryVariable boundary_variable =
        operator_data.bc_E->boundary_variable.find(boundary_id)->second;
      AssertThrow(boundary_variable != EnergyBoundaryVariable::Undefined,
                  ExcMessage("Energy boundary variable is undefined!"));

      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, true);

      VectorizedArray<value_type> tau_IP =
        fe_eval_momentum.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor) * nu;

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // density
        scalar rho_M      = fe_eval_density.get_value(q);
        scalar rho_inv_M  = 1.0 / rho_M;
        vector grad_rho_M = fe_eval_density.get_gradient(q);

        scalar rho_P =
          calculate_exterior_value<dim, value_type>(rho_M,
                                                    boundary_type_density,
                                                    operator_data.bc_rho,
                                                    boundary_id,
                                                    fe_eval_density.quadrature_point(q),
                                                    eval_time);

        scalar jump_density          = rho_M - rho_P;
        scalar gradient_flux_density = -tau_IP * jump_density;

        // velocity
        vector rho_u_M      = fe_eval_momentum.get_value(q);
        vector u_M          = rho_inv_M * rho_u_M;
        tensor grad_rho_u_M = fe_eval_momentum.get_gradient(q);

        vector u_P = calculate_exterior_value<dim, value_type>(u_M,
                                                               boundary_type_velocity,
                                                               operator_data.bc_u,
                                                               boundary_id,
                                                               fe_eval_momentum.quadrature_point(q),
                                                               eval_time);

        vector rho_u_P = rho_P * u_P;

        tensor grad_u_M = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
        tensor tau_M    = calculate_stress_tensor(grad_u_M, mu);

        vector tau_P_normal = calculate_exterior_normal_grad(tau_M * normal,
                                                             boundary_type_velocity,
                                                             operator_data.bc_u,
                                                             boundary_id,
                                                             fe_eval_momentum.quadrature_point(q),
                                                             eval_time);

        vector jump_momentum = rho_u_M - rho_u_P;
        vector gradient_flux_momentum =
          0.5 * (tau_M * normal + tau_P_normal) - tau_IP * jump_momentum;

        // energy
        scalar rho_E_M      = fe_eval_energy.get_value(q);
        scalar E_M          = rho_inv_M * rho_E_M;
        vector grad_rho_E_M = fe_eval_energy.get_gradient(q);

        scalar E_P = make_vectorized_array<value_type>(0.0);
        if(boundary_variable == EnergyBoundaryVariable::Energy)
        {
          E_P = calculate_exterior_value<dim, value_type>(E_M,
                                                          boundary_type_energy,
                                                          operator_data.bc_E,
                                                          boundary_id,
                                                          fe_eval_energy.quadrature_point(q),
                                                          eval_time);
        }
        else if(boundary_variable == EnergyBoundaryVariable::Temperature)
        {
          scalar p_M = calculate_pressure(rho_M, u_M, E_M, gamma);
          scalar T_M = calculate_temperature(p_M, rho_M, R);
          scalar T_P = calculate_exterior_value<dim, value_type>(T_M,
                                                                 boundary_type_energy,
                                                                 operator_data.bc_E,
                                                                 boundary_id,
                                                                 fe_eval_energy.quadrature_point(q),
                                                                 eval_time);

          value_type const c_v = R / (gamma - 1.0);

          E_P = calculate_energy(T_P, u_P, c_v);
        }
        scalar rho_E_P = rho_P * E_P;

        vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
        vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

        scalar grad_T_M_normal = grad_T_M * normal;
        scalar grad_T_P_normal =
          calculate_exterior_normal_grad<dim, value_type>(grad_T_M_normal,
                                                          boundary_type_energy,
                                                          operator_data.bc_E,
                                                          boundary_id,
                                                          fe_eval_energy.quadrature_point(q),
                                                          eval_time);

        scalar jump_energy          = rho_E_M - rho_E_P;
        scalar gradient_flux_energy = 0.5 * (u_M * tau_M * normal + u_P * tau_P_normal +
                                             lambda * (grad_T_M * normal + grad_T_P_normal)) -
                                      tau_IP * jump_energy;

        // value flux momentum
        vector jump_rho   = jump_density * normal;
        vector jump_rho_E = jump_energy * normal;
        tensor jump_rho_u = outer_product(jump_momentum, normal);

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
        vector value_flux_energy_M =
          -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

        fe_eval_density.submit_value(-gradient_flux_density, q);

        fe_eval_momentum.submit_gradient(value_flux_momentum_M, q);
        fe_eval_momentum.submit_value(-gradient_flux_momentum, q);

        fe_eval_energy.submit_gradient(value_flux_energy_M, q);
        fe_eval_energy.submit_value(-gradient_flux_energy, q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_momentum.integrate_scatter(true, true, dst);
      fe_eval_energy.integrate_scatter(true, true, dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  ViscousOperatorData<dim> operator_data;

  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;

  mutable value_type eval_time;
};

/*
 *  Combined operator: Evaluate viscous and convective term in one step to improve efficiency.
 *  The viscous operator already includes most of the terms. Additional terms needed by the
 *  convective term are highlighted by
 *
 *   // CONVECTIVE TERM
 *   implementations for convective term
 *   // CONVECTIVE TERM
 *
 */
template<int dim>
struct CombinedOperatorData
{
  CombinedOperatorData()
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

  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_rho;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_u;
  std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       bc_p;
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> bc_E;

  double dynamic_viscosity;
  double reference_density;
  double thermal_conductivity;
  double heat_capacity_ratio;
  double specific_gas_constant;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class CombinedOperator
{
public:
  typedef CombinedOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  CombinedOperator() : data(nullptr)
  {
  }

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>       FEEval_scalar;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEFaceEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type>     FEEval_vectorial;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEFaceEval_vectorial;

  typedef VectorizedArray<value_type>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<value_type>> vector;
  typedef Tensor<2, dim, VectorizedArray<value_type>> tensor;
  typedef Point<dim, VectorizedArray<value_type>>     point;

  void
  initialize(Mapping<dim> const &                mapping,
             MatrixFree<dim, value_type> const & mf_data,
             CombinedOperatorData<dim> const &   operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, fe_degree, value_type>(array_penalty_parameter,
                                                                *data,
                                                                mapping,
                                                                operator_data.dof_index);
  }

  void
  evaluate(parallel::distributed::Vector<value_type> &       dst,
           const parallel::distributed::Vector<value_type> & src,
           value_type const                                  evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, src, evaluation_time);
  }

  void
  evaluate_add(parallel::distributed::Vector<value_type> &       dst,
               const parallel::distributed::Vector<value_type> & src,
               value_type const                                  evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);

    // perform cell integrals only for performance measurements
    //    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void
  cell_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEval_scalar    fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.gather_evaluate(src, true, true);

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        scalar rho_inv  = 1.0 / fe_eval_density.get_value(q);
        vector grad_rho = fe_eval_density.get_gradient(q);

        vector rho_u      = fe_eval_momentum.get_value(q);
        vector u          = rho_inv * rho_u;
        tensor grad_rho_u = fe_eval_momentum.get_gradient(q);

        scalar rho_E      = fe_eval_energy.get_value(q);
        vector grad_rho_E = fe_eval_energy.get_gradient(q);

        // CONVECTIVE TERM
        tensor convective_flux = outer_product(rho_u, u);

        scalar p = calculate_pressure(rho_u, u, rho_E, gamma);
        for(unsigned int d = 0; d < dim; ++d)
          convective_flux[d][d] += p;
        // CONVECTIVE TERM

        // calculate flux momentum
        tensor grad_u = calculate_grad_u(rho_inv, rho_u, grad_rho, grad_rho_u);
        tensor tau    = calculate_stress_tensor(grad_u, mu);

        // calculate flux energy
        vector grad_E      = calculate_grad_E(rho_inv, rho_E, grad_rho, grad_rho_E);
        vector grad_T      = calculate_grad_T(grad_E, u, grad_u, gamma, R);
        vector flux_energy = tau * u + lambda * grad_T;

        fe_eval_density.submit_gradient(-rho_u /*CONV*/, q);
        fe_eval_momentum.submit_gradient(-convective_flux /*CONV*/ + tau /*VIS*/, q);
        fe_eval_energy.submit_gradient(-(rho_E + p) * u /*CONV*/ + flux_energy /*VIS*/, q);
      }

      fe_eval_density.integrate_scatter(false, true, dst);
      fe_eval_momentum.integrate_scatter(false, true, dst);
      fe_eval_energy.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop(const MatrixFree<dim, value_type> &               data,
            parallel::distributed::Vector<value_type> &       dst,
            const parallel::distributed::Vector<value_type> & src,
            const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_scalar fe_eval_density_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_vectorial fe_eval_momentum_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);
    FEFaceEval_scalar fe_eval_energy_neighbor(
      data, false, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const nu     = mu / operator_data.reference_density;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      // density
      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_density_neighbor.reinit(face);
      fe_eval_density_neighbor.gather_evaluate(src, true, true);

      // momentum
      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_momentum_neighbor.reinit(face);
      fe_eval_momentum_neighbor.gather_evaluate(src, true, true);

      // energy
      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, true);

      fe_eval_energy_neighbor.reinit(face);
      fe_eval_energy_neighbor.gather_evaluate(src, true, true);

      VectorizedArray<value_type> tau_IP =
        std::max(fe_eval_momentum.read_cell_data(array_penalty_parameter),
                 fe_eval_momentum_neighbor.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor) * nu;

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // density
        scalar rho_M      = fe_eval_density.get_value(q);
        scalar rho_inv_M  = 1.0 / rho_M;
        vector grad_rho_M = fe_eval_density.get_gradient(q);

        scalar rho_P      = fe_eval_density_neighbor.get_value(q);
        scalar rho_inv_P  = 1.0 / rho_P;
        vector grad_rho_P = fe_eval_density_neighbor.get_gradient(q);

        scalar jump_density          = rho_M - rho_P;
        scalar gradient_flux_density = -tau_IP * jump_density;

        // velocity
        vector rho_u_M      = fe_eval_momentum.get_value(q);
        vector u_M          = rho_inv_M * rho_u_M;
        tensor grad_rho_u_M = fe_eval_momentum.get_gradient(q);

        vector rho_u_P      = fe_eval_momentum_neighbor.get_value(q);
        vector u_P          = rho_inv_P * rho_u_P;
        tensor grad_rho_u_P = fe_eval_momentum_neighbor.get_gradient(q);

        tensor grad_u_M = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
        tensor tau_M    = calculate_stress_tensor(grad_u_M, mu);

        tensor grad_u_P = calculate_grad_u(rho_inv_P, rho_u_P, grad_rho_P, grad_rho_u_P);
        tensor tau_P    = calculate_stress_tensor(grad_u_P, mu);

        vector jump_momentum          = rho_u_M - rho_u_P;
        vector gradient_flux_momentum = 0.5 * (tau_M + tau_P) * normal - tau_IP * jump_momentum;

        // energy
        scalar rho_E_M      = fe_eval_energy.get_value(q);
        vector grad_rho_E_M = fe_eval_energy.get_gradient(q);
        scalar rho_E_P      = fe_eval_energy_neighbor.get_value(q);
        vector grad_rho_E_P = fe_eval_energy_neighbor.get_gradient(q);

        vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
        vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

        vector grad_E_P = calculate_grad_E(rho_inv_P, rho_E_P, grad_rho_P, grad_rho_E_P);
        vector grad_T_P = calculate_grad_T(grad_E_P, u_P, grad_u_P, gamma, R);

        vector flux_energy_average =
          0.5 * (tau_M * u_M + tau_P * u_P + lambda * (grad_T_M + grad_T_P));

        scalar jump_energy          = rho_E_M - rho_E_P;
        scalar gradient_flux_energy = flux_energy_average * normal - tau_IP * jump_energy;

        // value flux momentum
        vector jump_rho   = jump_density * normal;
        vector jump_rho_E = jump_energy * normal;
        tensor jump_rho_u = outer_product(jump_momentum, normal);

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
        vector value_flux_energy_M =
          -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

        vector grad_E_using_jumps_P = calculate_grad_E(rho_inv_P,
                                                       rho_E_P,
                                                       jump_rho /*instead of grad_rho*/,
                                                       jump_rho_E /*instead of grad_rho_E*/);

        vector grad_T_using_jumps_P =
          calculate_grad_T(grad_E_using_jumps_P, u_P, grad_u_using_jumps_P, gamma, R);
        vector value_flux_energy_P =
          -0.5 * (tau_using_jumps_P * u_P + lambda * grad_T_using_jumps_P);

        // CONVECTIVE TERM
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
        // CONVECTIVE TERM

        fe_eval_density.submit_value(flux_density /*CONV*/ - gradient_flux_density /*VIS*/, q);
        fe_eval_density_neighbor.submit_value(-flux_density /*CONV*/ +
                                                gradient_flux_density /*VIS*/,
                                              q);

        fe_eval_momentum.submit_gradient(value_flux_momentum_M /*VIS*/, q);
        fe_eval_momentum_neighbor.submit_gradient(value_flux_momentum_P /*VIS*/, q);

        fe_eval_momentum.submit_value(flux_momentum /*CONV*/ - gradient_flux_momentum /*VIS*/, q);
        fe_eval_momentum_neighbor.submit_value(-flux_momentum /*CONV*/ +
                                                 gradient_flux_momentum /*VIS*/,
                                               q);

        fe_eval_energy.submit_gradient(value_flux_energy_M /*VIS*/, q);
        fe_eval_energy_neighbor.submit_gradient(value_flux_energy_P /*VIS*/, q);

        fe_eval_energy.submit_value(flux_energy /*CONV*/ - gradient_flux_energy /*VIS*/, q);
        fe_eval_energy_neighbor.submit_value(-flux_energy /*CONV*/ + gradient_flux_energy /*VIS*/,
                                             q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_density_neighbor.integrate_scatter(true, false, dst);

      fe_eval_momentum.integrate_scatter(true, true, dst);
      fe_eval_momentum_neighbor.integrate_scatter(true, true, dst);

      fe_eval_energy.integrate_scatter(true, true, dst);
      fe_eval_energy_neighbor.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop(const MatrixFree<dim, value_type> &               data,
                     parallel::distributed::Vector<value_type> &       dst,
                     const parallel::distributed::Vector<value_type> & src,
                     const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEval_scalar fe_eval_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    value_type const gamma  = operator_data.heat_capacity_ratio;
    value_type const R      = operator_data.specific_gas_constant;
    value_type const mu     = operator_data.dynamic_viscosity;
    value_type const nu     = mu / operator_data.reference_density;
    value_type const lambda = operator_data.thermal_conductivity;

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      BoundaryType boundary_type_density = BoundaryType::undefined;
      if(operator_data.bc_rho->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_rho->dirichlet_bc.end())
      {
        boundary_type_density = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_rho->neumann_bc.find(boundary_id) !=
              operator_data.bc_rho->neumann_bc.end())
      {
        boundary_type_density = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_density != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      BoundaryType boundary_type_velocity = BoundaryType::undefined;
      if(operator_data.bc_u->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_u->dirichlet_bc.end())
      {
        boundary_type_velocity = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_u->neumann_bc.find(boundary_id) !=
              operator_data.bc_u->neumann_bc.end())
      {
        boundary_type_velocity = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_velocity != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      // CONVECTIVE TERM
      // pressure
      BoundaryType boundary_type_pressure = BoundaryType::undefined;
      if(operator_data.bc_p->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_p->dirichlet_bc.end())
      {
        boundary_type_pressure = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_p->neumann_bc.find(boundary_id) !=
              operator_data.bc_p->neumann_bc.end())
      {
        boundary_type_pressure = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_pressure != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));
      // CONVECTIVE TERM

      BoundaryType boundary_type_energy = BoundaryType::undefined;
      if(operator_data.bc_E->dirichlet_bc.find(boundary_id) !=
         operator_data.bc_E->dirichlet_bc.end())
      {
        boundary_type_energy = BoundaryType::dirichlet;
      }
      else if(operator_data.bc_E->neumann_bc.find(boundary_id) !=
              operator_data.bc_E->neumann_bc.end())
      {
        boundary_type_energy = BoundaryType::neumann;
      }
      AssertThrow(boundary_type_energy != BoundaryType::undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      EnergyBoundaryVariable boundary_variable =
        operator_data.bc_E->boundary_variable.find(boundary_id)->second;
      AssertThrow(boundary_variable != EnergyBoundaryVariable::Undefined,
                  ExcMessage("Energy boundary variable is undefined!"));

      fe_eval_density.reinit(face);
      fe_eval_density.gather_evaluate(src, true, true);

      fe_eval_momentum.reinit(face);
      fe_eval_momentum.gather_evaluate(src, true, true);

      fe_eval_energy.reinit(face);
      fe_eval_energy.gather_evaluate(src, true, true);

      VectorizedArray<value_type> tau_IP =
        fe_eval_momentum.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor) * nu;

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        vector normal = fe_eval_momentum.get_normal_vector(q);

        // density
        scalar rho_M      = fe_eval_density.get_value(q);
        scalar rho_inv_M  = 1.0 / rho_M;
        vector grad_rho_M = fe_eval_density.get_gradient(q);

        scalar rho_P =
          calculate_exterior_value<dim, value_type>(rho_M,
                                                    boundary_type_density,
                                                    operator_data.bc_rho,
                                                    boundary_id,
                                                    fe_eval_density.quadrature_point(q),
                                                    eval_time);

        scalar jump_density          = rho_M - rho_P;
        scalar gradient_flux_density = -tau_IP * jump_density;

        // velocity
        vector rho_u_M      = fe_eval_momentum.get_value(q);
        vector u_M          = rho_inv_M * rho_u_M;
        tensor grad_rho_u_M = fe_eval_momentum.get_gradient(q);

        vector u_P = calculate_exterior_value<dim, value_type>(u_M,
                                                               boundary_type_velocity,
                                                               operator_data.bc_u,
                                                               boundary_id,
                                                               fe_eval_momentum.quadrature_point(q),
                                                               eval_time);

        vector rho_u_P = rho_P * u_P;

        tensor grad_u_M = calculate_grad_u(rho_inv_M, rho_u_M, grad_rho_M, grad_rho_u_M);
        tensor tau_M    = calculate_stress_tensor(grad_u_M, mu);

        vector tau_P_normal = calculate_exterior_normal_grad(tau_M * normal,
                                                             boundary_type_velocity,
                                                             operator_data.bc_u,
                                                             boundary_id,
                                                             fe_eval_momentum.quadrature_point(q),
                                                             eval_time);

        vector jump_momentum = rho_u_M - rho_u_P;
        vector gradient_flux_momentum =
          0.5 * (tau_M * normal + tau_P_normal) - tau_IP * jump_momentum;

        // energy
        scalar rho_E_M      = fe_eval_energy.get_value(q);
        scalar E_M          = rho_inv_M * rho_E_M;
        vector grad_rho_E_M = fe_eval_energy.get_gradient(q);

        scalar E_P = make_vectorized_array<value_type>(0.0);
        if(boundary_variable == EnergyBoundaryVariable::Energy)
        {
          E_P = calculate_exterior_value<dim, value_type>(E_M,
                                                          boundary_type_energy,
                                                          operator_data.bc_E,
                                                          boundary_id,
                                                          fe_eval_energy.quadrature_point(q),
                                                          eval_time);
        }
        else if(boundary_variable == EnergyBoundaryVariable::Temperature)
        {
          scalar p_M = calculate_pressure(rho_M, u_M, E_M, gamma);
          scalar T_M = calculate_temperature(p_M, rho_M, R);
          scalar T_P = calculate_exterior_value<dim, value_type>(T_M,
                                                                 boundary_type_energy,
                                                                 operator_data.bc_E,
                                                                 boundary_id,
                                                                 fe_eval_energy.quadrature_point(q),
                                                                 eval_time);

          value_type const c_v = R / (gamma - 1.0);

          E_P = calculate_energy(T_P, u_P, c_v);
        }
        scalar rho_E_P = rho_P * E_P;

        vector grad_E_M = calculate_grad_E(rho_inv_M, rho_E_M, grad_rho_M, grad_rho_E_M);
        vector grad_T_M = calculate_grad_T(grad_E_M, u_M, grad_u_M, gamma, R);

        scalar grad_T_M_normal = grad_T_M * normal;
        scalar grad_T_P_normal =
          calculate_exterior_normal_grad<dim, value_type>(grad_T_M_normal,
                                                          boundary_type_energy,
                                                          operator_data.bc_E,
                                                          boundary_id,
                                                          fe_eval_energy.quadrature_point(q),
                                                          eval_time);

        scalar jump_energy          = rho_E_M - rho_E_P;
        scalar gradient_flux_energy = 0.5 * (u_M * tau_M * normal + u_P * tau_P_normal +
                                             lambda * (grad_T_M * normal + grad_T_P_normal)) -
                                      tau_IP * jump_energy;

        // value flux momentum
        vector jump_rho   = jump_density * normal;
        vector jump_rho_E = jump_energy * normal;
        tensor jump_rho_u = outer_product(jump_momentum, normal);

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
        vector value_flux_energy_M =
          -0.5 * (tau_using_jumps_M * u_M + lambda * grad_T_using_jumps_M);

        // CONVECTIVE TERM
        scalar p_M = calculate_pressure(rho_M, u_M, E_M, gamma);
        scalar p_P = calculate_exterior_value<dim, value_type>(p_M,
                                                               boundary_type_pressure,
                                                               operator_data.bc_p,
                                                               boundary_id,
                                                               fe_eval_density.quadrature_point(q),
                                                               eval_time);

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
        // CONVECTIVE_TERM

        fe_eval_density.submit_value(flux_density /*CONV*/ - gradient_flux_density /*VIS*/, q);

        fe_eval_momentum.submit_gradient(value_flux_momentum_M /*VIS*/, q);
        fe_eval_momentum.submit_value(flux_momentum /*CONV*/ - gradient_flux_momentum /*VIS*/, q);

        fe_eval_energy.submit_gradient(value_flux_energy_M /*VIS*/, q);
        fe_eval_energy.submit_value(flux_energy /*CONV*/ - gradient_flux_energy /*VIS*/, q);
      }

      fe_eval_density.integrate_scatter(true, false, dst);
      fe_eval_momentum.integrate_scatter(true, true, dst);
      fe_eval_energy.integrate_scatter(true, true, dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  CombinedOperatorData<dim> operator_data;

  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;

  mutable value_type eval_time;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_OPERATORS_H_ \
        */
