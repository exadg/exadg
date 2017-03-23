/*
 * PressureNeumannBCDivergenceTerm.h
 *
 *  Created on: Dec 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_DIVERGENCE_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_DIVERGENCE_TERM_H_



template<int dim>
class PressureNeumannBCDivergenceTermData
{
public:
  PressureNeumannBCDivergenceTermData()
    :
    dof_index_velocity(0),
    dof_index_pressure(0)
    {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};

template <int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class PressureNeumannBCDivergenceTerm: public BaseOperator<dim>
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

  typedef PressureNeumannBCDivergenceTerm<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  PressureNeumannBCDivergenceTerm()
    :
    data(nullptr),
    laplace_operator(nullptr)
  {}

  void initialize (MatrixFree<dim,value_type> const         &mf_data,
                   PressureNeumannBCDivergenceTermData<dim> &my_data_in,
                   LaplaceOperator<dim> const               &laplace_operator_in)
  {
    this->data = &mf_data;
    my_data = my_data_in;
    laplace_operator = &laplace_operator_in;
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
    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,my_data.dof_index_pressure);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,my_data.dof_index_velocity);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_pressure.reinit (face);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate (false,true);

      double factor = laplace_operator->get_penalty_factor();
      VectorizedArray<value_type> penalty_parameter = (value_type)factor *
          fe_eval_pressure.read_cell_data(laplace_operator->get_array_penalty_parameter());

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
      {
        it = my_data.bc->dirichlet_bc.find(boundary_id);
        if(it != my_data.bc->dirichlet_bc.end())
        {
          // do nothing on Dirichlet boundary Gamma^D = Neumann boundary Gamma^N_PPE for pressure
          fe_eval_pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
          fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
        }

        it = my_data.bc->neumann_bc.find(boundary_id);
        if (it != my_data.bc->neumann_bc.end())
        {
          VectorizedArray<value_type> div = fe_eval_velocity.get_divergence(q);

          fe_eval_pressure.submit_normal_gradient(-div,q); // minus sign since this term appears on the rhs of the equations
          fe_eval_pressure.submit_value(2.0 * penalty_parameter * div,q); // plus sign since this term appears on the rhs of the equations
        }
      }
      fe_eval_pressure.integrate(true,true);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  PressureNeumannBCDivergenceTermData<dim> my_data;

  LaplaceOperator<dim> const *laplace_operator;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_DIVERGENCE_TERM_H_ */
