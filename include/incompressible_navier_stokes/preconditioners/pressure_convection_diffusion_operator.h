/*
 * PressureConvectionDiffusionOperator.h
 *
 *  Created on: Jul 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_

#include "../../convection_diffusion/convection_diffusion_operators.h"
#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

template<int dim>
struct PressureConvectionDiffusionOperatorData
{
  PressureConvectionDiffusionOperatorData()
    :
    unsteady_problem(true),
    convective_problem(true)
  {}

  bool unsteady_problem;
  bool convective_problem;

  ScalarConvDiffOperators::MassMatrixOperatorData mass_matrix_operator_data;
  ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
  ScalarConvDiffOperators::ConvectiveOperatorDataDiscontinuousVelocity<dim> convective_operator_data;
};

template<int dim, int fe_degree, int fe_degree_velocity, typename value_type>
class PressureConvectionDiffusionOperator
{
public:
  PressureConvectionDiffusionOperator(Mapping<dim> const                                 &mapping,
                                      MatrixFree<dim,value_type> const                   &matrix_free_data_in,
                                      PressureConvectionDiffusionOperatorData<dim> const &operator_data_in)
    :
    matrix_free_data(matrix_free_data_in),
    operator_data(operator_data_in),
    scaling_factor_time_derivative_term(-1.0)
//    scaling_factor_time_derivative_term(nullptr)
  {
    // initialize MassMatrixOperator
    if(operator_data.unsteady_problem == true)
    {
      mass_matrix_operator.initialize(matrix_free_data,operator_data.mass_matrix_operator_data);
    }

    // initialize DiffusiveOperator
    diffusive_operator.initialize(mapping,matrix_free_data,operator_data.diffusive_operator_data);

    //initialize ConvectiveOperator
    if(operator_data.convective_problem == true)
    {
      convective_operator.initialize(matrix_free_data,operator_data.convective_operator_data);
    }
  }

  void apply(parallel::distributed::Vector<value_type>       &dst,
             parallel::distributed::Vector<value_type> const &src,
             parallel::distributed::Vector<value_type> const *velocity_vector)
  {
    // time derivate term in case of unsteady problems
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
          ExcMessage("Scaling factor of time derivative term has not been set for pressure convection-diffusion preconditioner!"));

      mass_matrix_operator.apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else // ensure that dst is initialized with 0.0 since diffusive operator calls apply_add and not apply
    {
      dst = 0.0;
    }

    // diffusive term
    diffusive_operator.apply_add(dst,src);

    // convective term
    if(operator_data.convective_problem == true)
    {
      AssertThrow(velocity_vector != nullptr, ExcMessage("velocity_vector is invalid."));

      convective_operator.apply_add(dst,src,velocity_vector);
    }
  }

  void set_scaling_factor_time_derivative_term(double const factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

private:
  MatrixFree<dim,value_type> const &matrix_free_data;
  PressureConvectionDiffusionOperatorData<dim> operator_data;
  double scaling_factor_time_derivative_term;
  ScalarConvDiffOperators::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
  ScalarConvDiffOperators::DiffusiveOperator<dim, fe_degree, value_type> diffusive_operator;
  ScalarConvDiffOperators::ConvectiveOperatorDiscontinuousVelocity<dim, fe_degree, fe_degree_velocity, value_type> convective_operator;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_ */
