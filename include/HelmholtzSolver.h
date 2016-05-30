/*
 * HelmholtzSolver.h
 *
 *  Created on: May 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_HELMHOLTZSOLVER_H_
#define INCLUDE_HELMHOLTZSOLVER_H_

//#include "FEEvaluationWrapper.h"
//#include "FE_Parameters.h"
//#include "InverseMassMatrix.h"

#include "PreconditionerVelocity.h"

template<int dim, typename value_type>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
    IP_formulation_viscous(InteriorPenaltyFormulationViscous::SIPG),
    IP_factor_viscous(1.0),
    dof_index(0)
  {}

  FormulationViscousTerm formulation_viscous_term;
  InteriorPenaltyFormulationViscous IP_formulation_viscous;
  double IP_factor_viscous;
  unsigned int dof_index;
  std::set<types::boundary_id> dirichlet_boundaries;
  std::set<types::boundary_id> neumann_boundaries;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall>
class HelmholtzOperator : public Subscriptor
{
public:
  HelmholtzOperator()
    :
    data(nullptr),
    fe_param(nullptr),
    mass_matrix_coefficient(-1.0),
    const_viscosity(-1.0)
  {}

  typedef double value_type;
  static const bool is_xwall = false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  void initialize(Mapping<dim> const & mapping,
                  MatrixFree<dim,value_type> const &mf_data,
                  FEParameters<value_type> & fe_param,
                  HelmholtzOperatorData<dim,value_type> const & operator_data)
  {
    this->data = &mf_data;
    this->fe_param = &fe_param;
    this->helmholtz_operator_data = operator_data;

    compute_array_penalty_parameter(mapping);
  }

  value_type get_penalty_factor() const
  {
    return helmholtz_operator_data.IP_factor_viscous * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  void set_mass_matrix_coefficient(double const coefficient_in)
  {
    mass_matrix_coefficient = coefficient_in;
  }

  void set_constant_viscosity(double const viscosity_in)
  {
    const_viscosity = viscosity_in;
  }

  void set_variable_viscosity(double const constant_viscosity_in)
  {
    const_viscosity = constant_viscosity_in;
    FEEval_Velocity_Velocity_linear fe_eval_velocity_cell(*data,*fe_param,helmholtz_operator_data.dof_index);
    viscous_coefficient_cell.reinit(data->n_macro_cells(), fe_eval_velocity_cell.n_q_points);
    viscous_coefficient_cell.fill(make_vectorized_array<value_type>(const_viscosity));

    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_face(*data,*fe_param,true,helmholtz_operator_data.dof_index);
    viscous_coefficient_face.reinit(data->n_macro_inner_faces()+data->n_macro_boundary_faces(), fe_eval_velocity_face.n_q_points);
    viscous_coefficient_face.fill(make_vectorized_array<value_type>(const_viscosity));

    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_face_neighbor(*data,*fe_param,false,helmholtz_operator_data.dof_index);
    viscous_coefficient_face_neighbor.reinit(data->n_macro_inner_faces()+data->n_macro_boundary_faces(), fe_eval_velocity_face.n_q_points);
    viscous_coefficient_face_neighbor.fill(make_vectorized_array<value_type>(const_viscosity));
  }

  // returns true if viscous_coefficient table has been filled with spatially varying viscosity values
  bool viscosity_is_variable() const
  {
    return viscous_coefficient_cell.n_elements()>0;
  }

  double get_const_viscosity() const
  {
    return const_viscosity;
  }

  Table<2,VectorizedArray<value_type> > const & get_viscous_coefficient_face() const
  {
    return viscous_coefficient_face;
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::BlockVector<double>       &dst,
              const parallel::distributed::BlockVector<double> &src) const
  {
    apply_viscous(src,dst);
  }

  void calculate_inverse_diagonal(parallel::distributed::BlockVector<value_type> &diagonal) const
  {
    diagonal = 0;

    parallel::distributed::BlockVector<value_type>  src_dummy(diagonal);

    data->loop(&HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_calculate_diagonal,
               &HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_calculate_diagonal_face,
               &HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_calculate_diagonal_boundary_face,
               this, diagonal, src_dummy);

    // verify calculation of diagonal
//    parallel::distributed::BlockVector<value_type>  diagonal2(diagonal);
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      diagonal2.block(d) = 0.0;
//    parallel::distributed::BlockVector<value_type>  src(diagonal2);
//    parallel::distributed::BlockVector<value_type>  dst(diagonal2);
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      for (unsigned int i=0;i<diagonal.block(d).local_size();++i)
//      {
//        src.block(d).local_element(i) = 1.0;
//        apply_viscous(src,dst);
//        diagonal2.block(d).local_element(i) = dst.block(d).local_element(i);
//        src.block(d).local_element(i) = 0.0;
//      }
//    //diagonal2.block(0).print(std::cout);
//
//    std::cout<<"L2 norm diagonal: "<<diagonal.l2_norm()<<std::endl;
//    std::cout<<"L2 norm diagonal2: "<<diagonal2.l2_norm()<<std::endl;
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      diagonal2.block(d).add(-1.0,diagonal.block(d));
//    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
    // verify calculation of diagonal

    //invert diagonal
    for(unsigned int d=0;d<diagonal.n_blocks();++d)
    {
      for (unsigned int i=0;i<diagonal.block(d).local_size();++i)
      {
        if( std::abs(diagonal.block(d).local_element(i)) > 1.0e-10 )
          diagonal.block(d).local_element(i) = 1.0/diagonal.block(d).local_element(i);
        else
          diagonal.block(d).local_element(i) = 1.0;
      }
    }
  }

private:
  void compute_array_penalty_parameter(const Mapping<dim> &mapping)
  {
    // Compute penalty parameter for each cell
    array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
    QGauss<dim> quadrature(fe_degree+1);
    FEValues<dim> fe_values(mapping,data->get_dof_handler(helmholtz_operator_data.dof_index).get_fe(),quadrature, update_JxW_values);
    QGauss<dim-1> face_quadrature(fe_degree+1);
    FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(helmholtz_operator_data.dof_index).get_fe(), face_quadrature, update_JxW_values);

    for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    {
      for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,helmholtz_operator_data.dof_index);
        fe_values.reinit(cell);
        double volume = 0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        double surface_area = 0;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_face_values.reinit(cell, f);
          const double factor = (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) ? 1. : 0.5;
          for (unsigned int q=0; q<face_quadrature.size(); ++q)
            surface_area += fe_face_values.JxW(q) * factor;
        }
        array_penalty_parameter[i][v] = surface_area / volume;
      }
    }
  }

  void apply_viscous (const parallel::distributed::BlockVector<value_type>  &src,
                      parallel::distributed::BlockVector<value_type>        &dst) const
  {
    for(unsigned int d=0;d<dim;++d)
    {
      dst.block(d)=0;
    }
    data->loop(&HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_apply_viscous,
               &HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_apply_viscous_face,
               &HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall>::local_apply_viscous_boundary_face,
               this, dst, src);
  }

  void local_apply_viscous (const MatrixFree<dim,value_type>                  &data,
                            parallel::distributed::BlockVector<double>        &dst,
                            const parallel::distributed::BlockVector<double>  &src,
                            const std::pair<unsigned int,unsigned int>        &cell_range) const
  {
    AssertThrow(mass_matrix_coefficient>0.0,ExcMessage("Mass matrix coefficient has not been set!"));
    AssertThrow(const_viscosity>0.0,ExcMessage("Constant viscosity has not been set!"));

    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,helmholtz_operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate (true,true);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        fe_eval_velocity.submit_value (mass_matrix_coefficient * fe_eval_velocity.get_value(q), q);

        VectorizedArray<value_type> viscosity = make_vectorized_array<value_type>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_cell[cell][q];

        if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          fe_eval_velocity.submit_gradient (viscosity*make_vectorized_array<value_type>(2.)*fe_eval_velocity.get_symmetric_gradient(q), q);
        }
        else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          fe_eval_velocity.submit_gradient (viscosity*fe_eval_velocity.get_gradient(q), q);
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
      }
      fe_eval_velocity.integrate (true,true);
      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  void local_apply_viscous_face (const MatrixFree<dim,value_type>                 &data,
                                 parallel::distributed::BlockVector<double>       &dst,
                                 const parallel::distributed::BlockVector<double> &src,
                                 const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,true,helmholtz_operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,*fe_param,false,helmholtz_operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity_neighbor.reinit (face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true,true);
      fe_eval_velocity_neighbor.read_dof_values(src,0,dim);
      fe_eval_velocity_neighbor.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval_velocity.read_cell_data(array_penalty_parameter),fe_eval_velocity_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
        VectorizedArray<value_type> average_viscosity = make_vectorized_array<value_type>(const_viscosity);
        VectorizedArray<value_type> max_viscosity = make_vectorized_array<value_type>(const_viscosity);
        if(viscosity_is_variable())
        {
          average_viscosity = 0.5*(viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
          max_viscosity = std::max(viscous_coefficient_face[face][q] , viscous_coefficient_face_neighbor[face][q]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor;
        if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
          average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
        }
        else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<value_type>(0.5);
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
        Tensor<2,dim,VectorizedArray<value_type> > jump_tensor =
            outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

        //we do not want to symmetrize the penalty part
        average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
        Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

        if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
          {
            fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            fe_eval_velocity_neighbor.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
          }
          else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
          {
            fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            fe_eval_velocity_neighbor.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
          }
          else
            AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
        }
        else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
          {
            fe_eval_velocity.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            fe_eval_velocity_neighbor.submit_gradient(0.5*average_viscosity*jump_tensor,q);
          }
          else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
          {
            fe_eval_velocity.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            fe_eval_velocity_neighbor.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
          }
          else
            AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
        fe_eval_velocity.submit_value(-average_gradient,q);
        fe_eval_velocity_neighbor.submit_value(average_gradient,q);
      }
      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
      fe_eval_velocity_neighbor.integrate(true,true);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst,0,dim);
    }
  }

  void local_apply_viscous_boundary_face (const MatrixFree<dim,value_type>                  &data,
                                          parallel::distributed::BlockVector<double>        &dst,
                                          const parallel::distributed::BlockVector<double>  &src,
                                          const std::pair<unsigned int,unsigned int>        &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,true,helmholtz_operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter)
                                             * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<value_type> viscosity = make_vectorized_array<value_type>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        if (helmholtz_operator_data.dirichlet_boundaries.find(data.get_boundary_indicator(face))
            != helmholtz_operator_data.dirichlet_boundaries.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM;
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor;
          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            average_gradient_tensor = make_vectorized_array<value_type>(2.) * fe_eval_velocity.get_symmetric_gradient(q);
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = fe_eval_velocity.get_gradient(q);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = viscosity*(average_gradient_tensor - jump_tensor * tau_IP);

          Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
        else if (helmholtz_operator_data.neumann_boundaries.find(data.get_boundary_indicator(face))
            != helmholtz_operator_data.neumann_boundaries.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
      }
      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
    }
  }

  void local_calculate_diagonal (const MatrixFree<dim,value_type>                  &data,
                                 parallel::distributed::BlockVector<double>        &dst,
                                 const parallel::distributed::BlockVector<double>  &,
                                 const std::pair<unsigned int,unsigned int>        &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,helmholtz_operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_velocity.evaluate (true,true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          // copied from local_apply_viscous //TODO
          fe_eval_velocity.submit_value (mass_matrix_coefficient * fe_eval_velocity.get_value(q), q);

          VectorizedArray<value_type> viscosity = make_vectorized_array<value_type>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_cell[cell][q];

          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            fe_eval_velocity.submit_gradient (viscosity*make_vectorized_array<value_type>(2.)*fe_eval_velocity.get_symmetric_gradient(q), q);
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            fe_eval_velocity.submit_gradient (viscosity*fe_eval_velocity.get_gradient(q), q);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          // copied from local_apply_viscous
        }
        fe_eval_velocity.integrate (true,true);
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j,local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_calculate_diagonal_face (const MatrixFree<dim,value_type>                 &data,
                                      parallel::distributed::BlockVector<double>       &dst,
                                      const parallel::distributed::BlockVector<double> &,
                                      const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,true,helmholtz_operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,*fe_param,false,helmholtz_operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval_velocity.read_cell_data(array_penalty_parameter),fe_eval_velocity_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));
        // set all dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

        fe_eval_velocity.evaluate(true,true);
        fe_eval_velocity_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          // copied from local_apply_viscous_face (note that fe_eval_neighbor.submit... has to be removed) //TODO
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
          VectorizedArray<value_type> average_viscosity = make_vectorized_array<value_type>(const_viscosity);
          VectorizedArray<value_type> max_viscosity = make_vectorized_array<value_type>(const_viscosity);
          if(viscosity_is_variable())
          {
            average_viscosity = 0.5*(viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
            max_viscosity = std::max(viscous_coefficient_face[face][q] , viscous_coefficient_face_neighbor[face][q]);
          }

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor;
          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
            average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<value_type>(0.5);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor =
              outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
          // copied from local_apply_viscous_face (note that fe_eval_neighbor.submit... has to be removed)
        }
        // integrate on element-
        fe_eval_velocity.integrate(true,true);
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j, local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global(dst);

      // neighbor (element+)
      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_velocity_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
      {
        // set all dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_velocity.evaluate(true,true);
        fe_eval_velocity_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          // copied from local_apply_viscous_face (note that fe_eval.submit... has to be removed)//TODO
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
          VectorizedArray<value_type> average_viscosity = make_vectorized_array<value_type>(const_viscosity);
          VectorizedArray<value_type> max_viscosity = make_vectorized_array<value_type>(const_viscosity);
          if(viscosity_is_variable())
          {
            average_viscosity = 0.5*(viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
            max_viscosity = std::max(viscous_coefficient_face[face][q] , viscous_coefficient_face_neighbor[face][q]);
          }

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor;
          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
            average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<value_type>(0.5);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor =
              outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            }
            else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity_neighbor.submit_value(average_gradient,q);
          // copied from local_apply_viscous_face  (note that fe_eval.submit... has to be removed)
        }
        // integrate on element+
        fe_eval_velocity_neighbor.integrate(true,true);
        local_diagonal_vector_neighbor[j] = fe_eval_velocity_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_calculate_diagonal_boundary_face (const MatrixFree<dim,value_type>                  &data,
                                               parallel::distributed::BlockVector<double>        &dst,
                                               const parallel::distributed::BlockVector<double>  &,
                                               const std::pair<unsigned int,unsigned int>        &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,*fe_param,true,helmholtz_operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter)
                                             * get_penalty_factor();

      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i, make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j, make_vectorized_array(1.));

        fe_eval_velocity.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          // copied from local_apply_viscous__boundary_face
          VectorizedArray<value_type> viscosity = make_vectorized_array<value_type>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_face[face][q];

          if (helmholtz_operator_data.dirichlet_boundaries.find(data.get_boundary_indicator(face))
              != helmholtz_operator_data.dirichlet_boundaries.end()) // Infow and wall boundaries
          {
            // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
            Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
            Tensor<1,dim,VectorizedArray<value_type> > uP = -uM;
            Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

            Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor;
            if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              average_gradient_tensor = make_vectorized_array<value_type>(2.) * fe_eval_velocity.get_symmetric_gradient(q);
            }
            else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              average_gradient_tensor = fe_eval_velocity.get_gradient(q);
            }
            else
            {
              AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
            }
            Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
              = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

            //we do not want to symmetrize the penalty part
            average_gradient_tensor = viscosity*(average_gradient_tensor - jump_tensor * tau_IP);

            Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

            if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              {
                fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              {
                fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric( viscosity*jump_tensor),q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              {
                fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
              }
              else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              {
                fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else
            {
              AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
            }
            fe_eval_velocity.submit_value(-average_gradient,q);
          }
          else if (helmholtz_operator_data.neumann_boundaries.find(data.get_boundary_indicator(face))
              != helmholtz_operator_data.neumann_boundaries.end()) // Outflow boundary
          {
            // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
            Tensor<1,dim,VectorizedArray<value_type> > jump_value;
            Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
            Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
              = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

            if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              {
                fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              {
                fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else if(helmholtz_operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              {
                fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
              }
              else if(helmholtz_operator_data.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              {
                fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else
            {
              AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
            }
            fe_eval_velocity.submit_value(-average_gradient,q);
          }
          // copied from local_apply_viscous__boundary_face
        }
        fe_eval_velocity.integrate(true,true);
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j, local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  FEParameters<value_type> const * fe_param;
  HelmholtzOperatorData<dim,value_type> helmholtz_operator_data;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double mass_matrix_coefficient;
  double const_viscosity;
  Table<2,VectorizedArray<value_type> > viscous_coefficient_cell;
  Table<2,VectorizedArray<value_type> > viscous_coefficient_face;
  Table<2,VectorizedArray<value_type> > viscous_coefficient_face_neighbor;
};

struct HelmholtzSolverData
{
  HelmholtzSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-12),
    solver_tolerance_rel(1.e-6),
    solver_viscous(SolverViscous::PCG),
    preconditioner_viscous(PreconditionerViscous::None)
    {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  SolverViscous solver_viscous;
  PreconditionerViscous preconditioner_viscous;
};


template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall>
class HelmholtzSolver
{
public:
  typedef double value_type;

  // Constructor.
  HelmholtzSolver()
    :
    global_matrix(nullptr),
    preconditioner(nullptr)
  {}
  // Destructor.
  ~HelmholtzSolver()
  {}

  void initialize(HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> const &helmholtz_operator,
                  HelmholtzSolverData const & solver_data_in,
                  MatrixFree<dim,value_type> const &mf_data,
                  const unsigned int dof_index,
                  const unsigned int quad_index)
  {
    this->global_matrix = &helmholtz_operator;
    this->solver_data = solver_data_in;

    if(solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
      preconditioner.reset(new InverseMassMatrixPreconditionerVelocity<dim,fe_degree,value_type>(mf_data,dof_index,quad_index));
    else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi)
      preconditioner.reset(new JacobiPreconditionerVelocity<dim,value_type, HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> >
                                (mf_data,dof_index,helmholtz_operator));
  }

  unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                     const parallel::distributed::BlockVector<double> &src) const;

private:
  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> const *global_matrix;
  HelmholtzSolverData solver_data;
  std_cxx11::shared_ptr<PreconditionerVelocityBase > preconditioner;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int HelmholtzSolver<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> ::
solve(parallel::distributed::BlockVector<double>       &dst,
      const parallel::distributed::BlockVector<double> &src) const
{
  ReductionControl solver_control (solver_data.max_iter,
                                   solver_data.solver_tolerance_abs,
                                   solver_data.solver_tolerance_rel);
  try
  {
    if(solver_data.solver_viscous == SolverViscous::PCG)
    {
      SolverCG<parallel::distributed::BlockVector<double> > solver (solver_control);
      if(solver_data.preconditioner_viscous == PreconditionerViscous::None)
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi ||
              solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
        solver.solve (*global_matrix, dst, src, *preconditioner);
    }
    else if(solver_data.solver_viscous == SolverViscous::GMRES)
    {
      SolverGMRES<parallel::distributed::BlockVector<double> > solver (solver_control);
      if(solver_data.preconditioner_viscous == PreconditionerViscous::None)
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi ||
              solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
        solver.solve (*global_matrix, dst, src, *preconditioner);
    }
    else
      AssertThrow(false,ExcMessage("Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
  }
  catch (SolverControl::NoConvergence &)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout << std::endl << "Viscous solver failed to solve to given tolerance." << std::endl;
  }

  return solver_control.last_step();
}

#endif /* INCLUDE_HELMHOLTZSOLVER_H_ */
