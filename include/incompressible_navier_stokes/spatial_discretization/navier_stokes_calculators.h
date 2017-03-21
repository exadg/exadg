/*
 * NavierStokesCalculators.h
 *
 *  Created on: Oct 28, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_

#include "../infrastructure/fe_evaluation_wrapper.h"
#include "operators/base_operator.h"

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class VorticityCalculator: public BaseOperator<dim>
{
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
public:
  VorticityCalculator()
  :
    data(nullptr),
    dof_index(0)
  {};

  void initialize (MatrixFree<dim,value_type> const &mf_data,
                   const unsigned int dof_index_in)
  {
    this->data = &mf_data;
    dof_index = dof_index_in;
  }

  void compute_vorticity (parallel::distributed::Vector<value_type>       &dst,
                          const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;

    data->cell_loop (&VorticityCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_compute_vorticity,this, dst, src);
  }

private:

  void local_compute_vorticity(const MatrixFree<dim,value_type>                 &data,
                               parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear velocity(data,this->fe_param,dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      velocity.reinit(cell);
      velocity.read_dof_values(src);
      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
        Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
        // omega_vector is a vector with dim components
        // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
        // for dim=2: omega_vector[0] = omega,
        //            omega_vector[1] = 0
        Tensor<1,dim,VectorizedArray<value_type> > omega_vector;
        for (unsigned int d=0; d<number_vorticity_components; ++d)
          omega_vector[d] = omega[d];
        velocity.submit_value (omega_vector, q);
      }
      velocity.integrate (true,false);
      velocity.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  unsigned int dof_index;
};


template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DivergenceCalculator: public BaseOperator<dim>
{
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
public:
  DivergenceCalculator()
  :
    data(nullptr),
    dof_index(0)
  {};

  void initialize (MatrixFree<dim,value_type> const &mf_data,
                   const unsigned int dof_index_in)
  {
    this->data = &mf_data;
    dof_index = dof_index_in;
  }

  void compute_divergence (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;

    data->cell_loop (&DivergenceCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_compute_divergence,this, dst, src);
  }

private:

  void local_compute_divergence(const MatrixFree<dim,value_type>                 &data,
                               parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(false,true);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
      {
        Tensor<1,dim,VectorizedArray<value_type> > div_vector;
          div_vector[0] = fe_eval_velocity.get_divergence(q);
        fe_eval_velocity.submit_value(div_vector,q);
      }
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  unsigned int dof_index;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_ */
