/*
 * PressureNeumannBCViscous.h
 *
 *  Created on: Nov 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_

#include "../../incompressible_navier_stokes/spatial_discretization/curl_compute.h"
#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_operators.h"
#include "../infrastructure/fe_evaluation_wrapper.h"
#include "operators/base_operator.h"

template<int dim>
class PressureNeumannBCViscousTermData
{
public:
  PressureNeumannBCViscousTermData()
    :
    dof_index_velocity(0),
    dof_index_pressure(0)
    {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};

template <int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class PressureNeumannBCViscousTerm: public BaseOperator<dim>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree_u+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEFaceEvaluationWrapper<dim,fe_degree_u,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall>
    FEFaceEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall>
    FEFaceEval_Pressure_Velocity_linear;

  typedef PressureNeumannBCViscousTerm<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  PressureNeumannBCViscousTerm()
    :
    data(nullptr),
    viscous_operator(nullptr)
  {}

  void initialize (MatrixFree<dim,value_type> const                   &mf_data,
                   PressureNeumannBCViscousTermData<dim>              &my_data_in,
                   ViscousOperator<dim, fe_degree_u, fe_degree_xwall,
                     xwall_quad_rule, value_type> const               &viscous_operator_in)
  {
    this->data = &mf_data;
    my_data = my_data_in;
    viscous_operator = &viscous_operator_in;
  }

  void calculate (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    this->data->loop(&This::cell_loop,&This::face_loop,&This::boundary_face_loop,this, dst, src);
  }

private:

  void cell_loop(const MatrixFree<dim,value_type>                &,
                 parallel::distributed::Vector<value_type>       &,
                 const parallel::distributed::Vector<value_type> &,
                 const std::pair<unsigned int,unsigned int>      &) const
  {

  }

  void face_loop(const MatrixFree<dim,value_type>                &,
                 parallel::distributed::Vector<value_type>       &,
                 const parallel::distributed::Vector<value_type> &,
                 const std::pair<unsigned int,unsigned int>      &) const
  {

  }

  void boundary_face_loop (const MatrixFree<dim,value_type>                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           const parallel::distributed::Vector<value_type>  &src,
                           const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_omega(data,this->fe_param,true,my_data.dof_index_velocity);

    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,my_data.dof_index_pressure);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_pressure.reinit (face);

      fe_eval_omega.reinit (face);
      fe_eval_omega.read_dof_values(src);
      fe_eval_omega.evaluate (false,true);

      typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
      {
        VectorizedArray<value_type> viscosity;
        if(this->viscous_operator->viscosity_is_variable())
          viscosity = this->viscous_operator->get_viscous_coefficient_face()[face][q];
        else
          viscosity = make_vectorized_array<value_type>(this->viscous_operator->get_const_viscosity());

        it = my_data.bc->dirichlet_bc.find(boundary_id);
        if(it != my_data.bc->dirichlet_bc.end())
        {
          VectorizedArray<value_type> h;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

          Tensor<1,dim,VectorizedArray<value_type> > curl_omega =
              CurlCompute<dim,FEFaceEvaluationWrapper<dim,fe_degree_u,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> >::compute(fe_eval_omega,q);
          h = - normal * (viscosity*curl_omega);

          fe_eval_pressure.submit_value(h,q);
        }

        it = my_data.bc->neumann_bc.find(boundary_id);
        if (it != my_data.bc->neumann_bc.end())
        {
          fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
        }
      }
      fe_eval_pressure.integrate(true,false);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  PressureNeumannBCViscousTermData<dim> my_data;
  ViscousOperator<dim, fe_degree_u, fe_degree_xwall, xwall_quad_rule, value_type>  const * viscous_operator;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_ */
