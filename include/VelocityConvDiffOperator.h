/*
 * VelocityConvDiffOperator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_VELOCITYCONVDIFFOPERATOR_H_
#define INCLUDE_VELOCITYCONVDIFFOPERATOR_H_


#include "NavierStokesOperators.h"

template<int dim>
struct VelocityConvDiffOperatorData
{
  VelocityConvDiffOperatorData ()
    :
    unsteady_problem(true),
    convective_problem(true)
  {}

  bool unsteady_problem;
  bool convective_problem;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall,typename Number = double>
class VelocityConvDiffOperator : public Subscriptor
{
public:
  typedef Number value_type;

  VelocityConvDiffOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    scaling_factor_time_derivative_term(nullptr),
    velocity_linearization(nullptr),
    evaluation_time(0.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                            &mf_data_in,
                  VelocityConvDiffOperatorData<dim> const                                                 &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> const     &viscous_operator_in,
                  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> const  &convective_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;
    this->convective_operator = &convective_operator_in;
  }

  void set_scaling_factor_time_derivative_term(Number const *factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  void set_velocity_linearization(parallel::distributed::Vector<Number> const *velocity_linearization_in)
  {
    velocity_linearization = velocity_linearization_in;
  }

  void set_evaluation_time(Number const &evaluation_time_in)
  {
    evaluation_time = evaluation_time_in;
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    AssertThrow(scaling_factor_time_derivative_term != nullptr,
      ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    if(operator_data.unsteady_problem == true)
    {
      mass_matrix_operator->apply(dst,src);
      dst *= (*scaling_factor_time_derivative_term);
    }
    else
    {
      dst = 0.0;
    }

    viscous_operator->apply_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_add(dst,src,velocity_linearization,evaluation_time);
    }
  }

private:
  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const * mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const * viscous_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> const * convective_operator;
  VelocityConvDiffOperatorData<dim> operator_data;
  Number const * scaling_factor_time_derivative_term;
  parallel::distributed::Vector<Number> const * velocity_linearization;
  Number evaluation_time;
};


#endif /* INCLUDE_VELOCITYCONVDIFFOPERATOR_H_ */
