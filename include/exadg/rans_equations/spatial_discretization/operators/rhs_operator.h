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
#include <exadg/rans_equations/spatial_discretization/turbulence_model.h>
#include <exadg/rans_equations/user_interface/viscosity_model_data.h>
#include <cmath>
#include "exadg/rans_equations/user_interface/enum_types.h"
#include "exadg/utilities/lazy_ptr.h"

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

  ScalarType                  scalar_type;
  bool                        production_term;
  bool                        dissipation_term;
  unsigned int                dof_index_eddy_viscosity;
  unsigned int                dof_index_velocity;
  unsigned int                dof_index;
  double                      diffusivity;
  TurbulenceModelData         turbulence_model_data;
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
    integrator_eddy_viscosity =
      std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);
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
   * Function for taking value of solution from pde_operator
   */
  void
  set_solution(VectorType const & sol)
  {
    // solution = &sol;
    solution.own() = sol;
    solution->update_ghost_values();
  }

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in)
  {
    eddy_viscosity.own() = eddy_viscosity_in;
    eddy_viscosity->update_ghost_values();
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
    integrator_solution->gather_evaluate(*solution,
                                         dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients);

    if(turbulence_model_ptr->turbulence_model_data.is_active)
    {
      integrator_eddy_viscosity->reinit(cell);
      integrator_eddy_viscosity->gather_evaluate(*eddy_viscosity, dealii::EvaluationFlags::values);
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

    if((data.turbulence_model_data.is_active))
    {
      scalar viscosity = integrator_eddy_viscosity->get_value(q);
      scalar dissipation;
      scalar production;
      if(data.dissipation_term)
      {
        f -= get_dissipation_scalar(viscosity, q);
      }
      if(data.production_term)
      {
        // AssertThrow(data.dissipation_term,
        //             dealii::ExcMessage("Production term only works with dissipation term"));
        // production =  std::min(get_production_scalar(viscosity, q), dealii::make_vectorized_array<Number>(10.0) * dissipation);
        f += get_production_scalar(viscosity, q);
      }
    }
    return f;
  }

  scalar
  get_production_scalar(scalar const & viscosity, unsigned int const q) const
  {
    scalar production;

    if(data.scalar_type == ScalarType::TurbulentKineticEnergy)
    {
      production = get_tke_production_term(viscosity, q);
    }
    else if(data.scalar_type == ScalarType::TKEDissipationRate)
    {
      production = get_epsilon_production_term(viscosity, q);
    }
    else
  {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Production term is only implemented for ScalarType::TurbulentKineticEnergy and ScalarType::TKEDissipationRate"));
    }
    return production;
  }

  scalar
  get_dissipation_scalar(scalar const & viscosity, unsigned int const q) const
  {
    scalar dissipation;
        if(data.scalar_type == ScalarType::TurbulentKineticEnergy)
        {
          dissipation = get_tke_dissipation_term(viscosity, q);
        }
        else if(data.scalar_type == ScalarType::TKEDissipationRate)
        {
          dissipation = get_epsilon_dissipation_term(viscosity, q);
        }
        else
        {
          AssertThrow(
            false,
            dealii::ExcMessage(
              "Dissipation term is only implemented for ScalarType::TurbulentKineticEnergy and ScalarType::TKEDissipationRate"));
        }
    return dissipation;
  }

  /**
   * 
   * \f[ Production = 2 \nu_{T} S_{ij} S_{ij} \f]
   ** For logarithmic variable:
   * \f[ Production = \frac{2 \nu_{T} S_{ij} S_{ij}}{k} \f]
   */
  scalar
  get_tke_production_term(scalar const & viscosity, unsigned int const q) const
  {
    scalar result;

    tensor velocity_gradient = integrator_velocity->get_gradient(q);

    tensor symmetric_velocity_gradient = (velocity_gradient + transpose(velocity_gradient));

    scalar gradient_product = scalar_product(symmetric_velocity_gradient, velocity_gradient);
    scalar sol           = integrator_solution->get_value(q);

    if(data.positivity_preserving_limiter ==
       PositivityPreservingLimiter::LogarithmicTransportVariable)
    {
      scalar tke = std::exp(sol);
      result = viscosity * gradient_product / tke;
    }
    else if(data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
    {
      result = viscosity * gradient_product;
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "PositivityPreservingLimiter needs to be specified for implementing tke production term"));
    }
    return result;
  }

  /*
   * \f[ Production = 2 \nu_{T} S_{ij} S_{ij} \f]
   * ** For logarithmic variable:
   * \f[ Production = \frac{2 \nu_{T} S_{ij} S_{ij}}{\epsilon} \f]
   */
  scalar
  get_epsilon_production_term(scalar const & viscosity, unsigned int const q) const
  {
    scalar result;

    tensor velocity_gradient = integrator_velocity->get_gradient(q);
    tensor symmetric_velocity_gradient = (velocity_gradient + transpose(velocity_gradient));

    scalar gradient_product = scalar_product(symmetric_velocity_gradient, velocity_gradient);

    scalar sol           = integrator_solution->get_value(q);

    scalar C_mu = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);
    scalar C_e1 = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);

    if(data.positivity_preserving_limiter ==
       PositivityPreservingLimiter::LogarithmicTransportVariable)
    {
      scalar epsilon = std::exp(sol);
      scalar tke = std::sqrt(viscosity * epsilon / C_mu);
      result = C_e1 * viscosity * gradient_product / tke;
    }
    else if(data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
    {
      scalar tke = std::sqrt(viscosity * sol / C_mu);
      result = C_e1 * viscosity * gradient_product;
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "PositivityPreservingLimiter needs to be specified for implementing epsilon production term"));
    }

    return result;
  }

  /** For Standard k-epsilon model:
   * \f[ \varepsilon = C_{\mu} \frac{e^{\kappa / 2}{\epsilon} \f]
   * For Standard k-epsilon model with logarithmic variable:
   * \f[ \varepsilon = C_{\mu} \frac{e^{\kappa}}{\nu_{T}} \f]
   */
  scalar
  get_tke_dissipation_term(scalar const & viscosity, unsigned int const q) const
  {
    scalar result;
    scalar sol = integrator_solution->get_value(q);

    if(turbulence_model_ptr->turbulence_model_data.turbulence_model ==
            TurbulenceEddyViscosityModel::StandardKEpsilon)
    {
      double C_mu = turbulence_model_ptr->model_coefficients[3]; // C_mu
      if(data.positivity_preserving_limiter ==
         PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        scalar viscosity_limit = std::max(viscosity, dealii::make_vectorized_array<Number>(1e-12));
        result = dealii::make_vectorized_array<Number>(C_mu) * std::exp(sol) / viscosity_limit;
      }
      else if(data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
      {
        result = dealii::make_vectorized_array<Number>(C_mu) * std::pow(sol, dealii::make_vectorized_array<Number>(2.0)) / viscosity;
      }
      else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "PositivityPreservingLimiter needs to be defined for implementing tke dissipation term"));
      }
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Dissipation term is only implemented for TurbulenceEddyViscosityModel::StandardKEpsilon"));
    }
    return result;
  }

  /** For Standard k-epsilon model:
   * \f[ Dissipation = C_{\epsilon 2} \frac{\varepsilon^{2}}{k} \f]
   * For Standard k-epsilon model with logarithmic variable:
   * \f[ Dissipation = C_{\epsilon 2} \sqrt{\frac{C_{\mu}}{\nu_{T}}} e^{\epsilon / 2} \f]
    *
   */
  scalar
  get_epsilon_dissipation_term(scalar const & viscosity, unsigned int const q) const
  {
    scalar result;
    scalar sol = integrator_solution->get_value(q);

    if(turbulence_model_ptr->turbulence_model_data.turbulence_model ==
       TurbulenceEddyViscosityModel::StandardKEpsilon)
    {
      scalar C_mu =
        dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);
      scalar C_e2 =
        dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);
      if(data.positivity_preserving_limiter ==
         PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        scalar epsilon = std::exp(sol);
        scalar tke = std::sqrt(viscosity * epsilon / C_mu);
        result = C_e2 * std::exp(sol) / tke;
      }
      else if(data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
      {
        scalar tke = std::sqrt(viscosity * sol / C_mu);
        result = C_e2 * std::pow(sol, dealii::make_vectorized_array<Number>(2.0)) / std::max(dealii::make_vectorized_array<Number>(1e-6), tke);
      }
      else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "PositivityPreservingLimiter needs to be defined for implementing epsilon dissipation term"));
      }
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Dissipation term for epsilon is only implemented for TurbulenceEddyViscosityModel::StandardKEpsilon"));
    }
    return result;
  }

private:
  mutable RHSKernelData<dim> data;

  mutable lazy_ptr<VectorType> velocity;

  std::shared_ptr<CellIntegratorVelocity> integrator_velocity;

  // const VectorType *solution;
  mutable lazy_ptr<VectorType> solution;

  mutable lazy_ptr<VectorType> eddy_viscosity;

  std::shared_ptr<IntegratorCell> integrator_solution;

  std::shared_ptr<IntegratorCell> integrator_eddy_viscosity;

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
  set_solution(VectorType const & src) const;

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in) const;

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
