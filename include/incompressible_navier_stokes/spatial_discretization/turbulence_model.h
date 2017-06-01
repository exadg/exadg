/*
 * turbulence_model.h
 *
 *  Created on: Apr 4, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

//#include "solvers_and_preconditioners/iterative_solvers.h"

// TODO

/*
 *  Operator data.
 */
struct ProjectionOperatorViscosityData
{
  ProjectionOperatorViscosityData()
    :
    penalty_parameter(1.0e0)
  {}

  double penalty_parameter;
};


// TODO

/*
 *  Projection operator using a continuity penalty term.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + (v_h, tau * [u_h])_dOmega^e
 *
 *  where v_h = test function, u_h = solution, tau = penalty parameter.
 *
 */
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorViscosity : public BaseOperator<dim>
{
public:
  typedef ProjectionOperatorViscosity<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> This;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
    dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
    dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  /*
   *  Constructor
   */
  ProjectionOperatorViscosity(MatrixFree<dim,value_type> const &data_in,
                              unsigned int const               dof_index_in,
                              unsigned int const               quad_index_in)
    :
    data(data_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    array_penalty_parameter(0)
  {
    array_penalty_parameter.resize(data.n_macro_cells()+data.n_macro_ghost_cells());

    calculate_array_penalty_parameter();
  }

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    apply(dst,src);
  }

  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;

    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src) const
  {
    this->data.loop(&This::cell_loop,&This::face_loop,
                    &This::boundary_face_loop, this, dst, src);
  }

  FEParameters<dim> const * get_fe_param() const
  {
    return this->fe_param;
  }


private:
  void calculate_array_penalty_parameter()
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
      }

      array_penalty_parameter[cell] = operator_data.penalty_parameter * std::exp(std::log(volume)/(double)dim);
    }
  }

  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      fe_eval.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_value (fe_eval.get_value(q), q);
      }
      fe_eval.integrate (true,false);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),true,dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->get_fe_param(),false,dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true,false);
      fe_eval_neighbor.evaluate(true,false);

      VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(array_penalty_parameter)
                                             + fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        fe_eval.submit_value(tau*jump_value,q);
        fe_eval_neighbor.submit_value(-tau*jump_value,q);
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop (const MatrixFree<dim,value_type>                &,
                           parallel::distributed::Vector<value_type>       &,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &) const
  {

  }


  MatrixFree<dim,value_type> const & data;
  unsigned int const dof_index;
  unsigned int const quad_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  ProjectionOperatorViscosityData operator_data;
};

/*
 *  Turbulence model data.
 */
struct TurbulenceModelData
{
  TurbulenceModelData()
    :
    turbulence_model(TurbulenceEddyViscosityModel::Undefined),
    constant(1.0)
  {}

  TurbulenceEddyViscosityModel turbulence_model;
  double constant;
};


/*
 *  Algebraic subgrid-scale turbulence models for LES of incompressible flows.
 */
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class TurbulenceModel
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall>
    FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall>
    FEFaceEval_Velocity_Velocity_linear;

  typedef TurbulenceModel<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number> This;

  /*
   *  Constructor.
   */
  TurbulenceModel()
    :
  matrix_free_data(nullptr),
  viscous_operator(nullptr)
  {}

  void initialize(MatrixFree<dim,Number> const                &matrix_free_data_in,
                  Mapping<dim> const                          &mapping,
                  ViscousOperator<dim, fe_degree,
                    fe_degree_xwall, xwall_quad_rule, Number> &viscous_operator_in,
                  TurbulenceModelData const                   &model_data)
  {
    matrix_free_data = &matrix_free_data_in;
    viscous_operator = &viscous_operator_in;

    calculate_filter_width(mapping);

    turb_model_data = model_data;
  }

  /*
   *  This function calculates the turbulent viscosity for a given velocity field.
   */
  void calculate_turbulent_viscosity(parallel::distributed::Vector<Number> const &velocity) const
  {
    parallel::distributed::Vector<Number> dummy;

    matrix_free_data->loop(&This::cell_loop_set_coefficients,
                           &This::face_loop_set_coefficients,
                           &This::boundary_face_loop_set_coefficients,
                           this, dummy, velocity);
  }

  void calculate_turbulent_viscosity(parallel::distributed::Vector<Number>       &dst,
                                     parallel::distributed::Vector<Number> const &velocity) const
  {
    // set dst-vector to zero
    dst = 0.0;

    // calculate rhs-vector
    matrix_free_data->cell_loop(&This::cell_loop_calculate_dof_vector, this, dst, velocity);
  }

private:
  void cell_loop_set_coefficients (MatrixFree<dim,Number> const                &data,
                                   parallel::distributed::Vector<Number>       &,
                                   parallel::distributed::Vector<Number> const &src,
                                   std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,
                                            viscous_operator->get_fe_param(),
                                            viscous_operator->get_operator_data().dof_index);

    // loop over all cells
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      // we only want to evaluate the gradient
      fe_eval.evaluate (false,true,false);

      // get filter width for this cell
      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient
        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);

        add_turbulent_viscosity(viscosity, filter_width, velocity_gradient, turb_model_data.constant);

        // set the coefficients
        viscous_operator->set_viscous_coefficient_cell(cell,q,viscosity);
      }
    }
  }

  void face_loop_set_coefficients (MatrixFree<dim,Number> const                &data,
                                   parallel::distributed::Vector<Number>       &,
                                   parallel::distributed::Vector<Number> const &src,
                                   std::pair<unsigned int,unsigned int> const  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                viscous_operator->get_fe_param(),
                                                true,
                                                viscous_operator->get_operator_data().dof_index);

    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,
                                                         viscous_operator->get_fe_param(),
                                                         false,
                                                         viscous_operator->get_operator_data().dof_index);

    // loop over all interior faces
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      // we only want to evaluate the gradient
      fe_eval.evaluate(false,true);
      fe_eval_neighbor.evaluate(false,true);

      // get filter width for this cell and the neighbor
      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);
      VectorizedArray<Number> filter_width_neighbor = fe_eval_neighbor.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());
        VectorizedArray<Number> viscosity_neighbor = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient for both elements adjacent to the current face
        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);
        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient_neighbor = fe_eval_neighbor.get_gradient(q);

        add_turbulent_viscosity(viscosity, filter_width, velocity_gradient, turb_model_data.constant);
        add_turbulent_viscosity(viscosity_neighbor, filter_width_neighbor, velocity_gradient_neighbor, turb_model_data.constant);

        // set the coefficients
        viscous_operator->set_viscous_coefficient_face(face,q,viscosity);
        viscous_operator->set_viscous_coefficient_face_neighbor(face,q,viscosity_neighbor);
      }
    }
  }

  // loop over all boundary faces
  void boundary_face_loop_set_coefficients (MatrixFree<dim,Number> const                &data,
                                            parallel::distributed::Vector<Number>       &,
                                            parallel::distributed::Vector<Number> const &src,
                                            std::pair<unsigned int,unsigned int> const  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                viscous_operator->get_fe_param(),
                                                true,
                                                viscous_operator->get_operator_data().dof_index);

    // loop over all boundary faces
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);

      // we only want to evaluate the gradient
      fe_eval.evaluate(false,true);

      // get filter width for this cell
      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient
        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);

        add_turbulent_viscosity(viscosity, filter_width, velocity_gradient, turb_model_data.constant);

        // set the coefficients
        viscous_operator->set_viscous_coefficient_face(face,q,viscosity);
      }
    }
  }

  void cell_loop_calculate_dof_vector (MatrixFree<dim,Number> const                &data,
                                       parallel::distributed::Vector<Number>       &dst,
                                       parallel::distributed::Vector<Number> const &src,
                                       std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,
                                            viscous_operator->get_fe_param(),
                                            viscous_operator->get_operator_data().dof_index);

    // loop over all cells
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      // we only want to evaluate the gradient
      fe_eval.evaluate (false,true,false);

      // get filter width for this cell
      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient
        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);

        add_turbulent_viscosity(viscosity, filter_width, velocity_gradient, turb_model_data.constant);

        Tensor<1,dim,VectorizedArray<Number> > vector;
        vector[0] = make_vectorized_array<Number>(1.0);

        fe_eval.submit_value (viscosity*vector, q);
      }
      fe_eval.integrate (true,false);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  /*
   *  This function calculates the filter width for each cell.
   */
  void calculate_filter_width(Mapping<dim> const &mapping)
  {
    filter_width_vector.resize(matrix_free_data->n_macro_cells()+matrix_free_data->n_macro_ghost_cells());

    unsigned int dof_index = viscous_operator->get_operator_data().dof_index;

    //QGauss<dim> quadrature(fe_degree+1);
    QGauss<dim> quadrature(fe_degree+1);

    FEValues<dim> fe_values(mapping,
                            matrix_free_data->get_dof_handler(dof_index).get_fe(),
                            quadrature,
                            update_JxW_values);

    // loop over all cells
    for (unsigned int i=0; i<matrix_free_data->n_macro_cells()+matrix_free_data->n_macro_ghost_cells(); ++i)
    {
      for (unsigned int v=0; v<matrix_free_data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = matrix_free_data->get_cell_iterator(i,v,dof_index);
        fe_values.reinit(cell);

        // calculate cell volume
        double volume = 0.0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
        {
          volume += fe_values.JxW(q);
        }

        // h = V^{1/dim}
        double h = std::exp(std::log(volume)/(double)dim);

        // take polynomial degree of shape functions into account:
        // TODO
        // h/(k_u + 1)
        h /= (double)(fe_degree+1);
        // h/(k_u)
        // h /= (double)(fe_degree);

        filter_width_vector[i][v] = h;
      }
    }
  }

  /*
   *  This function adds the turbulent eddy-viscosity to the laminar viscosity
   *  by using one of the implemented models.
   */
  void add_turbulent_viscosity (VectorizedArray<Number>                      &viscosity,
                                VectorizedArray<Number> const                &filter_width,
                                Tensor<2,dim,VectorizedArray<Number> > const &velocity_gradient,
                                double const                                 &model_constant) const
  {
    switch(turb_model_data.turbulence_model)
    {
      case TurbulenceEddyViscosityModel::Undefined:
        AssertThrow(turb_model_data.turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                    ExcMessage("parameter must be defined"));
        break;
      case TurbulenceEddyViscosityModel::Smagorinsky:
        smagorinsky_model(filter_width, velocity_gradient, model_constant, viscosity);
        break;
      case TurbulenceEddyViscosityModel::Vreman:
        vreman_model(filter_width, velocity_gradient, model_constant, viscosity);
        break;
      case TurbulenceEddyViscosityModel::WALE:
        wale_model(filter_width, velocity_gradient, model_constant, viscosity);
        break;
      case TurbulenceEddyViscosityModel::Sigma:
        sigma_model(filter_width, velocity_gradient, model_constant, viscosity);
        break;
    }
  }

  /*
   *  Smagorinsky model (1963):
   *
   *    nu_SGS = (C * filter_width)^{2} * sqrt(2 * S:S)
   *
   *    where S is the symmetric part of the velocity gradient
   *
   *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
   *
   *    and the model constant is
   *
   *      C = 0.165 (Nicoud et al. (2011))
   *      C = 0.18  (Toda et al. (2010))
   */
  void smagorinsky_model (VectorizedArray<Number> const                &filter_width,
                          Tensor<2,dim,VectorizedArray<Number> > const &velocity_gradient,
                          double const                                 &C,
                          VectorizedArray<Number>                      &viscosity) const
  {
    Tensor<2,dim,VectorizedArray<Number> > symmetric_gradient =
        make_vectorized_array<Number>(0.5)*(velocity_gradient + transpose(velocity_gradient));

    VectorizedArray<Number> rate_of_strain = 2.0*scalar_product(symmetric_gradient,symmetric_gradient);
    rate_of_strain = std::exp(0.5*std::log(rate_of_strain));

    VectorizedArray<Number> factor = C*filter_width;

    viscosity += factor * factor * rate_of_strain;
  }

  /*
   *  Vreman model (2004): Note that we only consider the isotropic variant of the Vreman model:
   *
   *    nu_SGS = (C * filter_width)^{2} * D
   *
   *  where the differential operator D is defined as
   *
   *    D = sqrt(B_gamma / ||grad(u)||^{2})
   *
   *  with
   *
   *    ||grad(u)||^{2} = grad(u) : grad(u) and grad(u) = d(u_i)/d(x_j) ,
   *
   *    gamma = grad(u) * grad(u)^T ,
   *
   *  and
   *
   *    B_gamma = gamma_11 * gamma_22 - gamma_12^{2}
   *             +gamma_11 * gamma_33 - gamma_13^{2}
   *             +gamma_22 * gamma_33 - gamma_23^{2}
   *
   *  Note that if ||grad(u)||^{2} = 0, nu_SGS is consistently defined as zero.
   *
   */
  void vreman_model (VectorizedArray<Number> const                &filter_width,
                     Tensor<2,dim,VectorizedArray<Number> > const &velocity_gradient,
                     double const                                 &C,
                     VectorizedArray<Number>                      &viscosity) const
  {
    VectorizedArray<Number> velocity_gradient_norm_square = scalar_product(velocity_gradient,velocity_gradient);
    Number const tolerance = 1.0e-12;

    VectorizedArray<bool> is_zero = make_vectorized_array<bool>(false);
    for(unsigned int i = 0; i < VectorizedArray<Number>::n_array_elements; i++)
    {
      if (velocity_gradient_norm_square[i] < tolerance)
      {
        is_zero[i] = true;
      }
    }

    Tensor<2,dim,VectorizedArray<Number> > tensor = velocity_gradient*transpose(velocity_gradient);

    AssertThrow(dim==3,ExcMessage("Number of dimensions has to be dim==3 to evaluate Vreman turbulence model."));

    VectorizedArray<Number> B_gamma =   tensor[0][0]*tensor[1][1] - tensor[0][1]*tensor[0][1]
                                      + tensor[0][0]*tensor[2][2] - tensor[0][2]*tensor[0][2]
                                      + tensor[1][1]*tensor[2][2] - tensor[1][2]*tensor[1][2];

    VectorizedArray<Number> factor = C*filter_width;

    for(unsigned int i = 0; i < VectorizedArray<Number>::n_array_elements; i++)
    {
      // If the norm of the velocity gradient tensor is zero (is_zero[i] == true),
      // the subgrid-scale viscosity is defined as zero, so we do nothing in that case.

      if (!is_zero[i])
      {
        // make sure that B_gamma[i] is larger than zero since we calculate the square root of B_gamma[i].
        Number const tolerance = 1.e-12;
        if(B_gamma[i] > tolerance)
        {
          viscosity[i] += factor[i] * factor[i] * std::exp(0.5*std::log(B_gamma[i]/velocity_gradient_norm_square[i]));
        }
      }
    }
  }

  /*
   *  WALE (wall-adapting local eddy-viscosity) model (Nicoud & Ducros 1999):
   *
   *    nu_SGS = (C * filter_width)^{2} * D ,
   *
   *  where the differential operator D is defined as
   *
   *    D = (S^{d}:S^{d})^{3/2} / ( (S:S)^{5/2} + (S^{d}:S^{d})^{5/4} )
   *
   *    where S is the symmetric part of the velocity gradient
   *
   *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
   *
   *    and S^{d} the traceless symmetric part of the square of the velocity
   *    gradient tensor
   *
   *      S^{d} = 1/2 * (g^{2} + (g^{2})^T) - 1/3 * trace(g^{2}) * I
   *
   *    with the square of the velocity gradient tensor
   *
   *      g^{2} = grad(u) * grad(u)
   *
   *    and the identity tensor I.
   *
   */
  void wale_model (VectorizedArray<Number> const                &filter_width,
                   Tensor<2,dim,VectorizedArray<Number> > const &velocity_gradient,
                   double const                                 &C,
                   VectorizedArray<Number>                      &viscosity) const
  {

    Tensor<2,dim,VectorizedArray<Number> > S =
        make_vectorized_array<Number>(0.5)*(velocity_gradient + transpose(velocity_gradient));
    VectorizedArray<Number> S_norm_square = scalar_product(S,S);

    Tensor<2,dim,VectorizedArray<Number> > square_gradient = velocity_gradient * velocity_gradient;
    VectorizedArray<Number> trace_square_gradient = trace(square_gradient);

    Tensor<2,dim,VectorizedArray<Number> >isotropic_tensor;
    for(unsigned int i=0;i<dim;++i)
    {
      isotropic_tensor[i][i] = 1.0/3.0 * trace_square_gradient;
    }

    Tensor<2,dim,VectorizedArray<Number> > S_d =
        make_vectorized_array<Number>(0.5)*(square_gradient + transpose(square_gradient)) - isotropic_tensor;
    VectorizedArray<Number> S_d_norm_square = scalar_product(S_d,S_d);

    VectorizedArray<Number> D = make_vectorized_array<Number>(0.0);

    for(unsigned int i = 0; i < VectorizedArray<Number>::n_array_elements; i++)
    {
      Number const tolerance = 1.e-12;
      if(S_d_norm_square[i] > tolerance)
      {
        D[i] = std::pow(S_d_norm_square[i],1.5)/(std::pow(S_norm_square[i],2.5) + std::pow(S_d_norm_square[i],1.25));
      }
    }

    VectorizedArray<Number> factor = C*filter_width;

    viscosity += factor * factor * D;
  }

  /*
   *  Sigma model (Toda et al. 2010, Nicoud et al. 2011):
   *
   *    nu_SGS = (C * filter_width)^{2} * D
   *
   *    where the differential operator D is defined as
   *
   *      D = s3 * (s1 - s2) * (s2 - s3) / s1^{2}
   *
   *    where s1 >= s2 >= s3 >= 0 are the singular values of
   *    the velocity gradient tensor g = grad(u).
   *
   *    The model constant is
   *
   *      C = 1.35 (Nicoud et al. (2011)) ,
   *      C = 1.5  (Toda et al. (2010)) .
   */
  void sigma_model (VectorizedArray<Number> const                &filter_width,
                    Tensor<2,dim,VectorizedArray<Number> > const &velocity_gradient,
                    double const                                 &C,
                    VectorizedArray<Number>                      &viscosity) const
  {
    /*
     *  Compute singular values manually using a self-contained method
     *  (see appendix in Nicoud et al. (2011)). This approach is more efficient
     *  than calculating eigenvalues or singular values using LAPACK routines.
     */
    VectorizedArray<Number> D = make_vectorized_array<Number>(0.0);

    Tensor<2,dim,VectorizedArray<Number> > G = transpose(velocity_gradient) * velocity_gradient;

    VectorizedArray<Number> invariant1 = trace(G);
    VectorizedArray<Number> invariant2 = 0.5*(invariant1*invariant1 - trace(G*G));
    VectorizedArray<Number> invariant3 = determinant(G);

    for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; n++)
    {
      // if trace(G) = 0, all eigenvalues (and all singular values) have to be zero
      // and hence G is also zero. Set D[n]=0 in that case.
      if(invariant1[n]>1.0e-12)
      {
        Number alpha1 = invariant1[n]*invariant1[n]/9.0 - invariant2[n]/3.0;
        Number alpha2 = invariant1[n]*invariant1[n]*invariant1[n]/27.0 - invariant1[n]*invariant2[n]/6.0 + invariant3[n]/2.0;
        AssertThrow(alpha1>= std::numeric_limits<double>::denorm_min() /*smallest positive value*/,
            ExcMessage("alpha1 has to be larger than zero."));

        Number factor = alpha2/std::pow(alpha1,1.5);
        AssertThrow(std::abs(factor)<=1.0+1.0e-12,
            ExcMessage("Cannot compute arccos(value) if abs(value)>1.0."));

        // Ensure that the argument of arccos() is in the interval [-1,1].
        if(factor > 1.0)
          factor = 1.0;
        else if(factor < -1.0)
          factor = -1.0;

        Number alpha3 = 1.0/3.0 * std::acos(factor);

        Vector<Number> sv = Vector<Number>(dim);

        sv[0] = invariant1[n]/3.0 + 2*std::sqrt(alpha1)*std::cos(alpha3);
        sv[1] = invariant1[n]/3.0 - 2*std::sqrt(alpha1)*std::cos(numbers::PI/3.0 + alpha3);
        sv[2] = invariant1[n]/3.0 - 2*std::sqrt(alpha1)*std::cos(numbers::PI/3.0 - alpha3);

        // Calculate the square root only if the value is larger than zero.
        // Otherwise set sv to zero (this is reasonable since negative values will
        // only occur due to numerical errors).
        for(unsigned int d=0;d<dim;++d)
        {
          if(sv[d] > 0.0)
            sv[d] = std::sqrt(sv[d]);
          else
            sv[d] = 0.0;
        }

        Number const tolerance = 1.e-12;
        if(sv[0] > tolerance)
        {
          D[n] = (sv[2]*(sv[0]-sv[1])*(sv[1]-sv[2]))/(sv[0]*sv[0]);
        }
      }
    }

    /*
     * The singular values of the velocity gradient g = grad(u)average_viscosity are
     * the square root of the eigenvalues of G = g^T * g.
     */

//    VectorizedArray<Number> D_copy = D; // save a copy in order to verify the correctness of the computation
//    for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; n++)
//    {
//      LAPACKFullMatrix<Number> G_local = LAPACKFullMatrix<Number>(dim);
//
//      for(unsigned int i = 0; i < dim; i++)
//      {
//        for(unsigned int j = 0; j < dim; j++)
//        {
//          G_local(i,j) = G[i][j][n];
//        }
//      }
//
//      G_local.compute_eigenvalues();
//
//      std::list<Number> ev_list;
//
//      for(unsigned int l = 0; l < dim; l++)
//      {
//        ev_list.push_back(std::abs(G_local.eigenvalue(l)));
//      }
//
//      // This sorts the list in ascending order, beginning with the smallest eigenvalue.
//      ev_list.sort();
//
//      Vector<Number> ev = Vector<Number>(dim);
//      typename std::list<Number>::reverse_iterator it;
//      unsigned int k;
//
//      // Write values in vector "ev" and reverse the order so that we
//      // ev[0] corresponds to the largest eigenvalue.
//      for(it = ev_list.rbegin(), k=0; it != ev_list.rend() && k<dim; ++it, ++k)
//      {
//        ev[k] = std::sqrt(*it);
//      }
//
//      Number const tolerance = 1.e-12;
//      if(ev[0] > tolerance)
//      {
//        D[n] = (ev[2]*(ev[0]-ev[1])*(ev[1]-ev[2]))/(ev[0]*ev[0]);
//      }
//    }
//
//    // make sure that both variants yield the same result
//    for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; n++)
//    {
//      AssertThrow(std::abs(D[n]-D_copy[n])<1.e-5,ExcMessage("Calculation of singular values is incorrect."));
//    }


    /*
     *  Alternatively, compute singular values directly using SVD.
     */
//    VectorizedArray<Number> D_copy2 = D; // save a copy in order to verify the correctness of the computation
//    D = make_vectorized_array<Number>(0.0);
//
//    for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; n++)
//    {
//      LAPACKFullMatrix<Number> gradient = LAPACKFullMatrix<Number>(dim);
//      for(unsigned int i = 0; i < dim; i++)
//      {
//        for(unsigned int j = 0; j < dim; j++)
//        {
//          gradient(i,j) = velocity_gradient[i][j][n];
//        }
//      }
//      gradient.compute_svd();
//
//      Vector<Number> sv = Vector<Number>(dim);
//      for(unsigned int i=0;i<dim;++i)
//      {
//        sv[i] = gradient.singular_value(i);
//      }
//
//      Number const tolerance = 1.e-12;
//      if(sv[0] > tolerance)
//      {
//        D[n] = (sv[2]*(sv[0]-sv[1])*(sv[1]-sv[2]))/(sv[0]*sv[0]);
//      }
//    }
//
//    // make sure that both variants yield the same result
//    for(unsigned int n = 0; n < VectorizedArray<Number>::n_array_elements; n++)
//    {
//      AssertThrow(std::abs(D[n]-D_copy2[n])<1.e-5,ExcMessage("Calculation of singular values is incorrect."));
//    }

    // add turbulent eddy-viscosity to laminar viscosity
    VectorizedArray<Number> factor = C*filter_width;
    viscosity += factor * factor * D;
  }


  TurbulenceModelData turb_model_data;

  MatrixFree<dim,Number> const * matrix_free_data;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> *viscous_operator;

  AlignedVector<VectorizedArray<Number> > filter_width_vector;
};




#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_ */
