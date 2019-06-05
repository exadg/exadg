/*
 * convection_diffusion_operator_efficiency.h
 *
 *  Created on: Nov 26, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_EFFICIENCY_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_EFFICIENCY_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/linear_operator_base.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/input_parameters.h"

#include "convection_diffusion/spatial_discretization/operators/diffusive_operator.h"
#include "convection_diffusion/spatial_discretization/operators/rhs_operator.h"
#include "convective_operator.h"
#include "functionalities/evaluate_functions.h"
#include "operators/interior_penalty_parameter.h"

namespace ConvDiff
{
// Convection-diffusion operator for runtime optimization:
// Evaluate volume and surface integrals of convective term, diffusive term and
// rhs term in one function (local_apply, local_apply_face, local_evaluate_boundary_face)
// instead of implementing each operator seperately and subsequently looping over all operators.
//
// Note: to obtain meaningful results, ensure that ...
//   ... the rhs-function, velocity-field and that the diffusivity is zero
//   if the rhs operator, convective operator or diffusive operator is "inactive".
//   The reason behind is that the volume and surface integrals of these operators
//   will always be evaluated for this "runtime optimization" implementation of the
//   convection-diffusion operator.
//
// Note: This operator is only implemented for the special case of explicit time integration,
//   i.e., when "evaluating" the operators for a given input-vector, at a given time and given
//   boundary conditions. Accordingly, the convective and diffusive operators are multiplied by
//   a factor of -1.0 since these terms are shifted to the right hand side of the equation.
//   The implicit solution of linear systems of equations (in case of implicit time integration)
//   is currently not available for this implementation.

template<int dim, typename Number>
struct ConvectionDiffusionOperatorDataEfficiency
{
  ConvectionDiffusionOperatorDataEfficiency()
  {
  }

  ConvectiveOperatorData<dim> conv_data;
  DiffusiveOperatorData<dim>  diff_data;
  RHSOperatorData<dim>        rhs_data;
};

template<int dim, typename Number>
class ConvectionDiffusionOperatorEfficiency
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ConvectionDiffusionOperatorEfficiency<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number> CellInt;
  typedef FaceIntegrator<dim, 1, Number> FaceInt;

  ConvectionDiffusionOperatorEfficiency() : data(nullptr), diffusivity(-1.0)
  {
  }

  void
  initialize(Mapping<dim> const &                                           mapping,
             MatrixFree<dim, Number> const &                                mf_data,
             ConvectionDiffusionOperatorDataEfficiency<dim, Number> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                                 *this->data,
                                                 mapping,
                                                 operator_data.diff_data.kernel_data.degree,
                                                 operator_data.diff_data.dof_index);

    diffusivity = operator_data.diff_data.kernel_data.diffusivity;
  }

  // Note: for this operator only the evaluate functions are implemented (no apply functions, no rhs
  // functions)

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

    data->loop(&This::local_apply_cell,
               &This::local_apply_face,
               &This::local_evaluate_boundary_face,
               this,
               dst,
               src);
  }

private:
  void
  local_apply_cell(MatrixFree<dim, Number> const &               data,
                   VectorType &                                  dst,
                   VectorType const &                            src,
                   std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellInt fe_eval(data, operator_data.diff_data.dof_index, operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.kernel_data.velocity->set_time(eval_time);
    operator_data.rhs_data.rhs->set_time(eval_time);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true, false);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

        // velocity
        Tensor<1, dim, VectorizedArray<Number>> velocity;
        for(unsigned int d = 0; d < dim; ++d)
        {
          Number array[VectorizedArray<Number>::n_array_elements];
          for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for(unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = operator_data.conv_data.kernel_data.velocity->value(q_point, d);
          }
          velocity[d].load(&array[0]);
        }

        // rhs
        VectorizedArray<Number> rhs = make_vectorized_array<Number>(0.0);

        Number array[VectorizedArray<Number>::n_array_elements];
        for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for(unsigned int d = 0; d < dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = operator_data.rhs_data.rhs->value(q_point);
        }
        rhs.load(&array[0]);

        fe_eval.submit_gradient(-1.0 * (-fe_eval.get_value(q) * velocity +
                                        make_vectorized_array<Number>(diffusivity) *
                                          fe_eval.get_gradient(q)),
                                q);
        // rhs term
        fe_eval.submit_value(rhs, q);
      }
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  local_apply_face(MatrixFree<dim, Number> const &               data,
                   VectorType &                                  dst,
                   VectorType const &                            src,
                   std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceInt fe_eval(data,
                    true,
                    operator_data.diff_data.dof_index,
                    operator_data.diff_data.quad_index);
    FaceInt fe_eval_neighbor(data,
                             false,
                             operator_data.diff_data.dof_index,
                             operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.kernel_data.velocity->set_time(eval_time);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true, true);

      VectorizedArray<Number> tau_IP =
        std::max(fe_eval.read_cell_data(array_penalty_parameter),
                 fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<Number>(operator_data.diff_data.kernel_data.degree,
                                       operator_data.diff_data.kernel_data.IP_factor);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

        Tensor<1, dim, VectorizedArray<Number>> velocity;
        for(unsigned int d = 0; d < dim; ++d)
        {
          Number array[VectorizedArray<Number>::n_array_elements];
          for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for(unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = operator_data.conv_data.kernel_data.velocity->value(q_point, d);
          }
          velocity[d].load(&array[0]);
        }
        Tensor<1, dim, VectorizedArray<Number>> normal = fe_eval.get_normal_vector(q);

        VectorizedArray<Number> u_n           = velocity * normal;
        VectorizedArray<Number> value_m       = fe_eval.get_value(q);
        VectorizedArray<Number> value_p       = fe_eval_neighbor.get_value(q);
        VectorizedArray<Number> average_value = 0.5 * (value_m + value_p);
        VectorizedArray<Number> jump_value    = value_m - value_p;
        VectorizedArray<Number> lambda        = std::abs(u_n);
        VectorizedArray<Number> lf_flux       = make_vectorized_array<Number>(0.0);

        if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
           NumericalFluxConvectiveOperator::CentralFlux)
        {
          lf_flux = u_n * average_value;
        }
        else if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
        {
          lf_flux = u_n * average_value + 0.5 * lambda * jump_value;
        }
        else
        {
          AssertThrow(
            this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                NumericalFluxConvectiveOperator::CentralFlux ||
              this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
            ExcMessage(
              "Specified numerical flux function for convective operator is not implemented!"));
        }

        VectorizedArray<Number> gradient_flux =
          (fe_eval.get_normal_derivative(q) + fe_eval_neighbor.get_normal_derivative(q)) * 0.5;
        gradient_flux = gradient_flux - tau_IP * jump_value;

        fe_eval.submit_normal_derivative(-1.0 * (-0.5 * diffusivity * jump_value), q);
        fe_eval_neighbor.submit_normal_derivative(-1.0 * (-0.5 * diffusivity * jump_value), q);

        fe_eval.submit_value(-1.0 * (lf_flux - diffusivity * gradient_flux), q);
        fe_eval_neighbor.submit_value(-1.0 * (-lf_flux + diffusivity * gradient_flux), q);
      }
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true, true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void
  local_evaluate_boundary_face(MatrixFree<dim, Number> const &               data,
                               VectorType &                                  dst,
                               VectorType const &                            src,
                               std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FaceInt fe_eval(data,
                    true,
                    operator_data.diff_data.dof_index,
                    operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.kernel_data.velocity->set_time(eval_time);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true);

      VectorizedArray<Number> tau_IP =
        fe_eval.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<Number>(operator_data.diff_data.kernel_data.degree,
                                       operator_data.diff_data.kernel_data.IP_factor);

      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      types::boundary_id boundary_id = data.get_boundary_id(face);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        it = operator_data.diff_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.diff_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
          // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
          // inhomogenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

          // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
          // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
          // inhomogeneous part: {{grad(u)}}*n = 0

          Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

          Tensor<1, dim, VectorizedArray<Number>> velocity;
          for(unsigned int d = 0; d < dim; ++d)
          {
            Number array[VectorizedArray<Number>::n_array_elements];
            for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = operator_data.conv_data.kernel_data.velocity->value(q_point, d);
            }
            velocity[d].load(&array[0]);
          }
          Tensor<1, dim, VectorizedArray<Number>> normal = fe_eval.get_normal_vector(q);

          VectorizedArray<Number> u_n     = velocity * normal;
          VectorizedArray<Number> value_m = fe_eval.get_value(q);

          // set the correct time for the evaluation of the boundary conditions
          it->second->set_time(eval_time);
          VectorizedArray<Number> g = make_vectorized_array<Number>(0.0);
          Number                  array[VectorizedArray<Number>::n_array_elements];
          for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for(unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          VectorizedArray<Number> value_p       = -value_m + 2.0 * g;
          VectorizedArray<Number> average_value = 0.5 * (value_m + value_p);
          VectorizedArray<Number> jump_value    = value_m - value_p;
          VectorizedArray<Number> lambda        = std::abs(u_n);
          VectorizedArray<Number> lf_flux       = make_vectorized_array<Number>(0.0);

          if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
             NumericalFluxConvectiveOperator::CentralFlux)
          {
            lf_flux = u_n * average_value;
          }
          else if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          {
            lf_flux = u_n * average_value + 0.5 * lambda * jump_value;
          }
          else
          {
            AssertThrow(
              this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::CentralFlux ||
                this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
              ExcMessage(
                "Specified numerical flux function for convective operator is not implemented!"));
          }

          VectorizedArray<Number> gradient_flux = fe_eval.get_normal_derivative(q);

          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_derivative(-1.0 * (-0.5 * diffusivity * jump_value), q);

          fe_eval.submit_value(-1.0 * (lf_flux - diffusivity * gradient_flux), q);
        }
        it = operator_data.diff_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.diff_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
          // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
          // inhomogenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

          // on GammaN: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
          // homogeneous part: {{grad(u)}}*n = 0
          // inhomogeneous part: {{grad(u)}}*n = h

          Point<dim, VectorizedArray<Number>>     q_points = fe_eval.quadrature_point(q);
          Tensor<1, dim, VectorizedArray<Number>> velocity;
          for(unsigned int d = 0; d < dim; ++d)
          {
            Number array[VectorizedArray<Number>::n_array_elements];
            for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = operator_data.conv_data.kernel_data.velocity->value(q_point, d);
            }
            velocity[d].load(&array[0]);
          }
          Tensor<1, dim, VectorizedArray<Number>> normal = fe_eval.get_normal_vector(q);

          VectorizedArray<Number> u_n           = velocity * normal;
          VectorizedArray<Number> value_m       = fe_eval.get_value(q);
          VectorizedArray<Number> value_p       = value_m;
          VectorizedArray<Number> average_value = 0.5 * (value_m + value_p);
          VectorizedArray<Number> jump_value    = value_m - value_p;
          VectorizedArray<Number> lambda        = std::abs(u_n);
          VectorizedArray<Number> lf_flux       = make_vectorized_array<Number>(0.0);

          if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
             NumericalFluxConvectiveOperator::CentralFlux)
          {
            lf_flux = u_n * average_value;
          }
          else if(this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          {
            lf_flux = u_n * average_value + 0.5 * lambda * jump_value;
          }
          else
          {
            AssertThrow(
              this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::CentralFlux ||
                this->operator_data.conv_data.kernel_data.numerical_flux_formulation ==
                  NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
              ExcMessage(
                "Specified numerical flux function for convective operator is not implemented!"));
          }

          VectorizedArray<Number> gradient_flux = make_vectorized_array<Number>(0.0);

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          Number array[VectorizedArray<Number>::n_array_elements];
          for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for(unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          gradient_flux.load(&array[0]);

          fe_eval.submit_normal_derivative(-1.0 * (-0.5 * diffusivity * jump_value), q);

          fe_eval.submit_value(-1.0 * (lf_flux - diffusivity * gradient_flux), q);
        }
      }
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  ConvectionDiffusionOperatorDataEfficiency<dim, Number> operator_data;

  AlignedVector<VectorizedArray<Number>> array_penalty_parameter;

  double diffusivity;

  mutable Number eval_time;
};

} // namespace ConvDiff



#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTION_DIFFUSION_OPERATOR_EFFICIENCY_H_ \
        */
