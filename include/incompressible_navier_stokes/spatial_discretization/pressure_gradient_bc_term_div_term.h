/*
 * PressureGradientBCTermDivTerm.h
 *
 *  Created on: Dec 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_GRADIENT_BC_TERM_DIV_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_GRADIENT_BC_TERM_DIV_TERM_H_



template<int dim>
class PressureGradientBCTermDivTermData
{
public:
  PressureGradientBCTermDivTermData()
    :
    dof_index_velocity(0)
    {}

  unsigned int dof_index_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};

template <int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class PressureGradientBCTermDivTerm: public BaseOperator<dim>
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

  typedef PressureGradientBCTermDivTerm<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  PressureGradientBCTermDivTerm()
    :
    data(nullptr)
  {}

  void initialize (MatrixFree<dim,value_type> const       &mf_data,
                   PressureGradientBCTermDivTermData<dim> &my_data_in)
  {
    this->data = &mf_data;
    my_data = my_data_in;
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

  void  face_loop(const MatrixFree<dim,value_type>                &,
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
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,my_data.dof_index_velocity);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate (false,true);

      typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        it = my_data.bc->dirichlet_bc.find(boundary_id);
        if(it != my_data.bc->dirichlet_bc.end())
        {
          // do nothing on Dirichlet boundary Gamma^D = Neumann boundary Gamma^N_PPE for pressure
          Tensor<1,dim,VectorizedArray<value_type> > normal;
          fe_eval_velocity.submit_value(normal,q);
        }

        it = my_data.bc->neumann_bc.find(boundary_id);
        if (it != my_data.bc->neumann_bc.end())
        {
          VectorizedArray<value_type> div = fe_eval_velocity.get_divergence(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);

          fe_eval_velocity.submit_value(div*normal,q);
        }
      }
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  PressureGradientBCTermDivTermData<dim> my_data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_GRADIENT_BC_TERM_DIV_TERM_H_ */
