/*
 * turbulence_model.h
 *
 *  Created on: Apr 4, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

struct TurbulenceModelData
{
  TurbulenceModelData()
    :
    constant(1.0)
  {}

  double constant;
};

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

    matrix_free_data->loop(&This::cell_loop,
                           &This::face_loop,
                           &This::boundary_face_loop,
                           this, dummy, velocity);
  }

private:
  void cell_loop (const MatrixFree<dim,Number>                 &data,
                  parallel::distributed::Vector<Number>        &,
                  const parallel::distributed::Vector<Number>  &src,
                  const std::pair<unsigned int,unsigned int>   &cell_range) const
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
//      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient
//        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);

        // TODO: calculate turbulent viscosity according to turbulence model
        // viscosity += ...;

        // set the coefficients
        viscous_operator->set_viscous_coefficient_cell(cell,q,viscosity);
      }
    }
  }

  void face_loop (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
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
//      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);
//      VectorizedArray<Number> filter_width_neighbor = fe_eval_neighbor.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());
        VectorizedArray<Number> viscosity_neighbor = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient for both elements adjacent to the current face
//        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);
//        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient_neighbor = fe_eval_neighbor.get_gradient(q);

        // TODO: calculate turbulent viscosity according to turbulence model

        // set the coefficients
        viscous_operator->set_viscous_coefficient_face(face,q,viscosity);
        viscous_operator->set_viscous_coefficient_face_neighbor(face,q,viscosity_neighbor);
      }
    }
  }

  // loop over all boundary faces
  void boundary_face_loop (const MatrixFree<dim,Number>                 &data,
                           parallel::distributed::Vector<Number>        &,
                           const parallel::distributed::Vector<Number>  &src,
                           const std::pair<unsigned int,unsigned int>   &face_range) const
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
//      VectorizedArray<Number> filter_width = fe_eval.read_cell_data(this->filter_width_vector);

      // loop over all quadrature points
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(viscous_operator->get_const_viscosity());

        // calculate velocity gradient
//        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);

        // TODO: calculate turbulent viscosity according to turbulence model

        // set the coefficients
        viscous_operator->set_viscous_coefficient_face(face,q,viscosity);
      }
    }
  }

  /*
   *  This function calculates the filter width for each cell.
   */
  void calculate_filter_width(Mapping<dim> const &mapping)
  {
    filter_width_vector.resize(matrix_free_data->n_macro_cells()+matrix_free_data->n_macro_ghost_cells());

    unsigned int dof_index = viscous_operator->get_operator_data().dof_index;

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

        // take polynomial degree of shape functions
        // into account: -> h/(k_u + 1)
        h /= (double)(fe_degree+1);

        filter_width_vector[i][v] = h;
      }
    }
  }

  TurbulenceModelData turb_model_data;

  MatrixFree<dim,Number> const * matrix_free_data;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> *viscous_operator;

  AlignedVector<VectorizedArray<Number> > filter_width_vector;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_ */
