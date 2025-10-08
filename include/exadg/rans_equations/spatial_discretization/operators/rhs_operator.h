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

#ifndef RANS_EQUATIONS_RHS_OPERATOR
#define RANS_EQUATIONS_RHS_OPERATOR

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

#include <exadg/operators/operator_base.h>
#include <exadg/rans_equations/user_interface/parameters.h>
#include <algorithm>
#include <memory>
// #include <exadg/utilities/lazy_ptr.h>
#include <cmath>
#include "exadg/rans_equations/user_interface/enum_types.h"
#include "exadg/utilities/lazy_ptr.h"
#include <exadg/rans_equations/spatial_discretization/turbulence_model.h>
#include <exadg/rans_equations/user_interface/viscosity_model_data.h>

namespace ExaDG
{
namespace RANS
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  RHSKernelData() : scalar_type(ScalarType::Scalar)
  {
  }
  std::shared_ptr<dealii::Function<dim>> f;

  ScalarType   scalar_type;
  bool         production_term;
  bool         dissipation_term;
  unsigned int dof_index_velocity;
  unsigned int dof_index;
  TurbulenceModelData turbulence_model_data;
  PositivityPreservingLimiter positivity_preserving_limiter;
};

template<int dim, typename Number, int n_components = 1>
class RHSKernel
{
private:
  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef dealii::VectorizedArray<Number>   scalar;
  typedef dealii::Tensor<rank, dim, scalar> value;


  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef CellIntegrator<dim, dim, Number>                   CellIntegratorVelocity;
  typedef dealii::Tensor<1, dim, scalar>                     vector;
  typedef dealii::Tensor<2, dim, scalar>                     tensor;
  typedef dealii::SymmetricTensor<2, dim, scalar>            symmertric_tensor;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free_in,
         RHSKernelData<dim> const &              data_in,
         unsigned int const                      quad_index)
  {
    data = data_in;
    integrator_velocity =
      std::make_shared<CellIntegratorVelocity>(matrix_free_in, data.dof_index_velocity, quad_index);
    integrator_solution =
      std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index, quad_index);
    integrator_rans_secondary_variable =
      std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index, quad_index);
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values |
                  dealii::update_quadrature_points; // q-points due to rhs function f

    // no face integrals

    return flags;
  }

  /*
   * Function for taking value of velocity from NS solver
   */
  void
  set_velocity_ptr(VectorType const & velocity_in)
  {
    velocity.own() = velocity_in;
    velocity->update_ghost_values();
  }

  /*
   * Function for taking value of velocity from NS solver
   */
  void
  set_rans_secondary_variable_ptr(VectorType const & src)
  {
    rans_secondary_variable.own() = src;
    rans_secondary_variable->update_ghost_values();
  }

  /*
   * Function for taking value of solution from pde_operator
   */
  void
  set_solution(VectorType const & sol)
  {
    // solution = &sol;
    solution.own() = sol;
    solution->update_ghost_values();
  }

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_body_force_production() const
  {
    return production_scalar;
  }

  /*
   * Function for initialising FEEvaluation class of with velocity
   */
  void
  reinit_cell_velocity(unsigned int const cell) const
  {
    integrator_velocity->reinit(cell);
    integrator_velocity->gather_evaluate(*velocity,
                                         dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients);
  }

  void
  reinit_cell_solution(unsigned int const cell) const
  {
    integrator_solution->reinit(cell);
    integrator_solution->gather_evaluate(*solution, dealii::EvaluationFlags::values);

    if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      if (data.scalar_type==ScalarType::TKEDissipationRate) {
        integrator_rans_secondary_variable->reinit(cell);
        integrator_rans_secondary_variable->gather_evaluate(*rans_secondary_variable, dealii::EvaluationFlags::values);
      }
    }
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(IntegratorCell const & integrator,
                    unsigned int const     q,
                    Number const &         time) const
  {
    dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

    scalar f = FunctionEvaluator<rank, dim, Number>::value(*(data.f), q_points, time);

    if((data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::PrandtlMixingLengthModel) || data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
    {
        scalar viscosity = turbulence_model_ptr->get_viscosity_cell(integrator.get_current_cell_index(), q, VaryingViscosityType::EddyViscosity);
        scalar dissipation;
        scalar production;
      if(data.dissipation_term)
      {
        if (data.scalar_type==ScalarType::TurbulentKineticEnergy) {
          dissipation = get_tke_dissipation_term(viscosity, q);
        }
        else if (data.scalar_type==ScalarType::TKEDissipationRate) {
          dissipation = get_epsilon_dissipation_term(viscosity, q);
        }
        else {
          AssertThrow(false, dealii::ExcMessage("Dissipation term is only implemented for ScalarType::TurbulentKineticEnergy and ScalarType::TKEDissipationRate"));
        }
        f -= dissipation;
      }
      if(data.production_term)
      {
        AssertThrow(data.dissipation_term, dealii::ExcMessage("Production term only works with dissipation term"));
        if (data.scalar_type==ScalarType::TurbulentKineticEnergy) {
          production = get_tke_production_term(viscosity, q);
        }
        else if (data.scalar_type==ScalarType::TKEDissipationRate) {
          production = get_epsilon_production_term(viscosity, q);
        }
        else {
          AssertThrow(false, dealii::ExcMessage("Production term is only implemented for ScalarType::TurbulentKineticEnergy and ScalarType::TKEDissipationRate"));
        }
        production = std::min(production, dealii::make_vectorized_array<Number>(10.0) * dissipation);
        f += production;
      }
    }

    return f;
  }

  /*
   * Production = 2 \nu_{T} S_{ij} S_{ij} / e^{\kappa}
   */
  scalar
  get_tke_production_term(scalar const & viscosity,
                          unsigned int const q) const
  {
    scalar result;

    tensor velocity_gradient = integrator_velocity->get_gradient(q);

    tensor velocity_gradient_tensor = 0.5 * (velocity_gradient + transpose(velocity_gradient));

    scalar forb_norm_sqr = velocity_gradient_tensor.norm_square();
    scalar sol = integrator_solution->get_value(q);

    if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
      result = dealii::make_vectorized_array<Number>(2.0) * viscosity * forb_norm_sqr / std::exp(sol);
    }
    else if(data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper){
      result = dealii::make_vectorized_array<Number>(2.0) * viscosity * forb_norm_sqr;
    }
    else {
      AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for implementing tke production term"));
    }

    /*std::cout << "TKE Production : " << result << std::endl;*/

    return result;
  }

  /*
   * Production = 2 \nu_{T} S_{ij} S_{ij} / e^{\kappa}
   */
  scalar
  get_epsilon_production_term(scalar const & viscosity,
                                  unsigned int const q) const
  {
    scalar result;

    tensor velocity_gradient = integrator_velocity->get_gradient(q);

    tensor velocity_gradient_tensor = 0.5 * (velocity_gradient + transpose(velocity_gradient));

    scalar forb_norm_sqr = velocity_gradient_tensor.norm_square();
    scalar sol = integrator_solution->get_value(q);


    if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
      result = dealii::make_vectorized_array<Number>(2.0 * turbulence_model_ptr->k_epsilon_data_base->C_epsilon_1) * viscosity * forb_norm_sqr / std::exp(sol);
    }
    else if(data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper) {
      result = dealii::make_vectorized_array<Number>(2.0 * turbulence_model_ptr->k_epsilon_data_base->C_epsilon_1) * viscosity * forb_norm_sqr;
    }
    else {
      AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for implementing epsilon production term"));
    }

    /*std::cout << "Epsilon Production : " << result << std::endl;*/
    
    return result;
  }

  /*
   * \varepsilon = - C_{D} \frac{e^{\kappa / 2}{\ell}}
   */
  scalar
  get_tke_dissipation_term(scalar const & viscosity,
                           unsigned int const q) const
  {
    scalar result;
    scalar sol = integrator_solution->get_value(q);
    if(turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLengthModel)
    {
    double coefficient  = turbulence_model_ptr->prandtl_mixing_length_data_base->C_D;
    double length_scale = turbulence_model_ptr->prandtl_mixing_length_data_base->turbulent_length_scale;
      if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
        result = dealii::make_vectorized_array<Number>(coefficient/length_scale) * std::exp(sol/2.0);
      }
      else if(data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper){
        scalar sol_max = dealii::make_vectorized_array<Number>(0.0);
        sol_max = std::max(sol, sol_max);
        result = dealii::make_vectorized_array<Number>(coefficient/length_scale) * std::pow(sol_max, dealii::make_vectorized_array<Number>(1.5));
      }
      else {
        AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be defined for implementing tke dissipation term"));
      }
    }
    else if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      double C_mu = turbulence_model_ptr->k_epsilon_data_base->C_mu;
      if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
        result = dealii::make_vectorized_array<Number>(C_mu) * std::exp(sol) / viscosity;
      }
      else if(data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper){
        result = dealii::make_vectorized_array<Number>(C_mu) * sol * sol / viscosity;
      }
      else {
        AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be defined for implementing tke dissipation term"));
      }
    }
    else {
    AssertThrow(false, dealii::ExcMessage("Dissipation term is only implemented for TurbulenceEddyViscosityModel::PrandtlMixingLengthModel"));
    }
    /*std::cout << "TKE Dissipation : " << result << std::endl;*/
    return result;
  }

  /*
   * \varepsilon = - C_{D} \frac{e^{\kappa / 2}{\ell}}
   */
  scalar
  get_epsilon_dissipation_term(scalar const & viscosity,
                               unsigned int const q) const
  {
    scalar result;
    scalar sol = integrator_solution->get_value(q);
    scalar tke = integrator_rans_secondary_variable->get_value(q);
    if (turbulence_model_ptr->turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      double C_epsilon_2 = turbulence_model_ptr->k_epsilon_data_base->C_epsilon_2;
      if (data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
        result = dealii::make_vectorized_array<Number>(C_epsilon_2) * std::exp(sol - tke);
      }
      else if(data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper){
        result = dealii::make_vectorized_array<Number>(C_epsilon_2) * sol * sol / tke;
      }
      else {
        AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be defined for implementing epsilon dissipation term"));
      }
    }
    else {
      AssertThrow(false, dealii::ExcMessage("Dissipation term for epsilon is only implemented for TurbulenceEddyViscosityModel::StandardKEpsilon"));
    }
    /*std::cout << "Epsilon Dissipation : " << result << std::endl;*/
    return result;
  }

private:
  mutable RHSKernelData<dim> data;

  mutable lazy_ptr<VectorType> velocity;

  std::shared_ptr<CellIntegratorVelocity> integrator_velocity;

  // const VectorType *solution;
  mutable lazy_ptr<VectorType> solution;

  mutable lazy_ptr<VectorType> rans_secondary_variable;

  std::shared_ptr<IntegratorCell> integrator_solution;

  std::shared_ptr<IntegratorCell> integrator_rans_secondary_variable;

  scalar production_scalar;

  Number prod_max, prod_min;

public:
  std::shared_ptr<TurbulenceModel<dim, Number>> turbulence_model_ptr;
};

} // namespace Operators


template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  Operators::RHSKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class RHSOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef RHSOperator<dim, Number, n_components> This;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * Constructor.
   */
  RHSOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &                          matrix_free,
             RHSOperatorData<dim> const &                                     data,
             std::shared_ptr<Operators::RHSKernel<dim, Number, n_components>> kernel);

  void
  set_velocity_ptr(VectorType const & velocity_in) const;

  void
  set_rans_secondary_variable_ptr(VectorType const & src) const;

  void
  set_solution(VectorType const & src) const;

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  // Operators::RHSKernel<dim, Number, n_components> kernel;
  std::shared_ptr<Operators::RHSKernel<dim, Number, n_components>> kernel;
};

} // namespace RANS
} // namespace ExaDG

#endif
