/*
 * pressure_convection_diffusion_operator.h
 *
 *  Created on: Jul 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_

#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../../convection_diffusion/spatial_discretization/operators/convective_operator.h"
#include "../../convection_diffusion/spatial_discretization/operators/diffusive_operator.h"
#include "../../convection_diffusion/spatial_discretization/operators/mass_operator.h"

namespace IncNS
{
template<int dim>
struct PressureConvectionDiffusionOperatorData
{
  PressureConvectionDiffusionOperatorData() : unsteady_problem(true), convective_problem(true)
  {
  }

  bool unsteady_problem;
  bool convective_problem;

  ConvDiff::MassMatrixOperatorData      mass_matrix_operator_data;
  ConvDiff::DiffusiveOperatorData<dim>  diffusive_operator_data;
  ConvDiff::ConvectiveOperatorData<dim> convective_operator_data;
};

template<int dim, typename value_type>
class PressureConvectionDiffusionOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  PressureConvectionDiffusionOperator(
    Mapping<dim> const &                                 mapping,
    MatrixFree<dim, value_type> const &                  matrix_free_data_in,
    PressureConvectionDiffusionOperatorData<dim> const & operator_data_in,
    AffineConstraints<double> &                          constraint_matrix)
    : matrix_free_data(matrix_free_data_in),
      operator_data(operator_data_in),
      scaling_factor_time_derivative_term(-1.0)
  {
    (void)mapping;
    // initialize MassMatrixOperator
    if(operator_data.unsteady_problem == true)
    {
      mass_matrix_operator.reinit(matrix_free_data,
                                  constraint_matrix,
                                  operator_data.mass_matrix_operator_data);
    }

    // initialize DiffusiveOperator
    diffusive_operator.reinit(matrix_free_data,
                              constraint_matrix,
                              operator_data.diffusive_operator_data);

    // initialize ConvectiveOperator
    if(operator_data.convective_problem == true)
    {
      convective_operator.reinit(matrix_free_data,
                                 constraint_matrix,
                                 operator_data.convective_operator_data);
    }
  }

  void
  apply(VectorType & dst, VectorType const & src, VectorType const & velocity_vector)
  {
    // time derivate term in case of unsteady problems
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(
        scaling_factor_time_derivative_term > 0.0,
        ExcMessage(
          "Scaling factor of time derivative term has not been set for pressure convection-diffusion preconditioner!"));

      mass_matrix_operator.apply(dst, src);
      dst *= scaling_factor_time_derivative_term;
    }
    else // ensure that dst is initialized with 0.0 since diffusive operator calls apply_add and not
         // apply
    {
      dst = 0.0;
    }

    // diffusive term
    diffusive_operator.apply_add(dst, src);

    // convective term
    if(operator_data.convective_problem == true)
    {
      convective_operator.set_velocity(velocity_vector);
      convective_operator.apply_add(dst, src);
    }
  }

  void
  set_scaling_factor_time_derivative_term(double const factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

private:
  MatrixFree<dim, value_type> const &          matrix_free_data;
  PressureConvectionDiffusionOperatorData<dim> operator_data;

  ConvDiff::MassMatrixOperator<dim, value_type> mass_matrix_operator;
  ConvDiff::DiffusiveOperator<dim, value_type>  diffusive_operator;
  ConvDiff::ConvectiveOperator<dim, value_type> convective_operator;

  double scaling_factor_time_derivative_term;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_PRESSURE_CONVECTION_DIFFUSION_OPERATOR_H_ \
        */
